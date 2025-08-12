import os
import boto3 # type: ignore
import json
import time
import datetime
import warnings
import hashlib
import torch
import pickle
import pandas as pd
import numpy as np
import awsgi # type: ignore
import joblib
import redis # type: ignore
import redis.cluster # type: ignore
from redis.cluster import ClusterNode # type: ignore
from flask import Flask, request, jsonify # type: ignore
from flask_cors import cross_origin
from waitress import serve
from dotenv import load_dotenv # type: ignore

import src.model.torch_model as t
import src.data_handling as data_handling
from src._utils import main_logger, s3_load, s3_load_to_temp_file

# silence warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# load env
AWS_LAMBDA_RUNTIME_API = os.environ.get('AWS_LAMBDA_RUNTIME_API', None)
if AWS_LAMBDA_RUNTIME_API is None: load_dotenv(override=True)


# file path
PRODUCTION_MODEL_FOLDER_PATH = 'models/production'
DFN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'dfn_best.pth')
GBM_FILE_PATH =  os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'gbm_best.pth')
EN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'en_best.pth')

PREPROCESSOR_PATH = 'preprocessors/column_transformer.pkl'
ORIGINAL_DF_PATH = os.environ.get('ORIGINAL_DF_PATH')

# s3 boto client
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-sales-pred')
s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'))

try:
    # test boto3 client
    sts_client = boto3.client('sts')
    identity = sts_client.get_caller_identity()
    main_logger.info(f"✅ Lambda is using role: {identity['Arn']}")

    # test s3 access
    response = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=PREPROCESSOR_PATH)
    main_logger.info("✅ S3 access works")

except Exception as e:
    main_logger.error(f"❌ Lambda credentials/permissions error: {e}")


# flask app config
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CLIENT_A =  os.environ.get('CLIENT_A')
API_ENDPOINT = os.environ.get('API_ENDPOINT')
origins = ['http://localhost:3000', CLIENT_A, API_ENDPOINT]


# global variables
_redis_client = None
preprocessor = None
model = None
backup_model = None


def get_redis_client():
    global _redis_client

    if _redis_client is None:
        REDIS_HOST = os.environ.get("REDIS_HOST")
        REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
        REDIS_TLS = os.environ.get("REDIS_TLS", "true").lower() == "true"


        if not REDIS_HOST:
            main_logger.error("REDIS_HOST environment variable not set. Redis connection will fail.")

        try:
            startup_nodes = [ClusterNode(host=REDIS_HOST, port=REDIS_PORT)]
            _redis_client = redis.cluster.RedisCluster(
                startup_nodes=startup_nodes,
                # password=REDIS_PASSWORD,
                decode_responses=True,
                skip_full_coverage_check=True,
                ssl=REDIS_TLS,                  # elasticache has encryption in transit: enabled -> must be true
                ssl_cert_reqs=None,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
                retry_on_timeout=True,
                retry_on_error=[
                    redis.exceptions.ConnectionError,
                    redis.exceptions.TimeoutError
                ],
                max_connections=10,            # limit connections for Lambda
                max_connections_per_node=2     # limit per node
            )
            _redis_client.ping()
            main_logger.info("successfully connected to ElastiCache Redis Cluster (Configuration Endpoint)")
        except redis.exceptions.ConnectionError as e:
            main_logger.error(f"could not connect to Redis Cluster: {e}", exc_info=True)
            _redis_client = None
        except Exception as e:
            main_logger.error(f"an unexpected error occurred during Redis Cluster connection: {e}", exc_info=True)
            _redis_client = None
    return _redis_client


def load_preprocessor():
    global preprocessor

    main_logger.info("... loading transformer ...")
    try:
        preprocessor_tempfile_path = s3_load_to_temp_file(PREPROCESSOR_PATH)
        if preprocessor_tempfile_path:
            preprocessor = joblib.load(preprocessor_tempfile_path)
            os.remove(preprocessor_tempfile_path)
        else:
            preprocessor = joblib.load(PREPROCESSOR_PATH)

    except:
        preprocessor = joblib.load(PREPROCESSOR_PATH)


def load_artifacts_primary_model():
    global model

    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)

    try:
        # load trained processor
        main_logger.info("... loading artifacts - trained dfn ...")
        input_dim = 63
        model_data_bytes_io = s3_load(file_path=DFN_FILE_PATH)

        if model_data_bytes_io is not None:
            state_dict = torch.load(model_data_bytes_io, weights_only=False, map_location=device)
            model = t.scripts.load_model(input_dim=input_dim, trig='best', saved_state_dict=state_dict)
            model.eval()
        else:
            model = t.scripts.load_model(input_dim=input_dim, trig='best')
            model.eval()

    except Exception as e:
        main_logger.critical(f"failed to load one or more essential artifacts during cold start: {e}", exc_info=True)
        try:
            model = t.scripts.load_model(file_path=DFN_FILE_PATH)
        except:
            raise RuntimeError("flask application failed to initialize due to missing or invalid model artifacts.")


def load_artifacts_backup_model():
    global backup_model
    model_data_bytes_io = None

    try:
        main_logger.info('... loading gbm ...')
        model_data_bytes_io = s3_load(file_path=GBM_FILE_PATH)
        if model_data_bytes_io:
            model_data_bytes_io.seek(0)
            loaded_dict = pickle.load(model_data_bytes_io)
            backup_model = loaded_dict['best_model']
            main_logger.info("... successfully loaded gbm ...")
            return

    except:
        main_logger.error(f"failed to load gbm from s3 using pickle. try loading elastic net instead as a backup model.")
        try:
            main_logger.info('... loading en ...')
            model_data_bytes_io = s3_load(file_path=EN_FILE_PATH)
            if model_data_bytes_io:
                model_data_bytes_io.seek(0)
                loaded_dict = pickle.load(model_data_bytes_io)
                backup_model = loaded_dict['best_model']
                main_logger.info("... successfully loaded elastic net ...")
                return
        except Exception as e:
            main_logger.critical(f"failed to load elastic net from s3 using pickle: {e}")


# api endpoints
@app.route('/')
def hello_world():
    env = os.environ.get('ENV', None)
    if env == 'local': return """<p>Hello, world</p><p>I am an API endpoint.</p>"""

    data = request.json if request.is_json else request.data.decode('utf-8')
    main_logger.info(f"request received! ENV: {env}, Data: {data}")
    return jsonify({"message": f"Hello from Flask in Lambda! ENV: {env}", "received_data": data})


@app.route('/v1/predict-price/<string:stockcode>', methods=['GET', 'OPTIONS'])
@cross_origin(origins=origins, methods=['GET', 'OPTIONS'], supports_credentials=True)
def predict_price(stockcode):
    df_stockcode = None

    try:
        main_logger.info('... start to predict ...')

        data = request.args.to_dict() # can incl. country, unitprice_min, unitprice_max, customerid, num_price_bins
        stockcode = stockcode if stockcode else data.get('stockcode', '85123A') if data is not None else '85123A'
        main_logger.info(f'predicting for stockcode: {stockcode}')
        main_logger.info(f'query parameters: {data}')

        # define cache key for df_stockcode
        cache_key_df_stockcode, file_path_df_stockcode = data_handling.scripts.fetch_imputation_cache_key_and_file_path(stockcode=stockcode)
        hash_data_st = { 'stockcode': stockcode  }
        params_hash_st = hashlib.sha256(str(sorted(hash_data_st.items())).encode()).hexdigest()
        cache_key_df_stockcode = f'{cache_key_df_stockcode}:{params_hash_st}'

        # define cache key for prediction
        hash_data = { 'stockcode': stockcode, **data }
        params_hash = hashlib.sha256(str(sorted(hash_data.items())).encode()).hexdigest()
        cache_key_prediction_result_by_stockcode = f"prediction:{{{stockcode}}}:{params_hash}"

        # fetch cache
        if _redis_client is not None:
            # prediction results
            cached_prediction_result = _redis_client.get(cache_key_prediction_result_by_stockcode)
            if cached_prediction_result:
                main_logger.info(f"cached prediction hit for stockcode: {stockcode}")
                return jsonify(json.loads(cached_prediction_result))

            # load historical data of the product (stockcode)
            cached_df_stockcode = _redis_client.get(cache_key_df_stockcode)
            if cached_df_stockcode: df_stockcode = json.loads(cached_df_stockcode)

        if df_stockcode is None:
            df_stockcode_bites_io = s3_load(
                file_path=file_path_df_stockcode[1:] if file_path_df_stockcode[0] == '/' else file_path_df_stockcode
            )
            if df_stockcode_bites_io: df_stockcode = pd.read_parquet(df_stockcode_bites_io)

        # define the price range
        min_price, max_price = data.get('unitprice_min', None), data.get('unitprice_max', None)

        if min_price is None:
            if df_stockcode is not None and not df_stockcode.empty:
                min_price = df_stockcode['unitprice_min'][0]
                if min_price < 0.1: min_price = df_stockcode['unitprice_median'][0]
            else:
                min_price = 2

        if max_price is None:
            if df_stockcode is not None and not df_stockcode.empty:
                max_price = df_stockcode['unitprice_max'][0]
            else:
                max_price = 20

        if min_price == max_price:
            min_price = max(2, min_price * 0.9)
            max_price = max_price * 1.5 + 0.1
        elif min_price > max_price:
            min_price, max_price = max_price, min_price

        if not 'unitprice_max' in data and max_price - min_price < 10:
            max_price = max_price * 3

        NUM_PRICE_BINS = data.get('num_price_bins', 5000)
        price_range = np.linspace(min_price, max_price, NUM_PRICE_BINS)
        main_logger.info(f'price ranges for stockcode {stockcode}: ${min_price} - ${max_price}')

        # impute input data
        customerid = data.get('customerid', 'unknown') if data else 'unknown'
        try: customer_recency_days = df_stockcode.loc[df_stockcode['customerid'] == customerid, 'customer_recency_days_latest'].iloc[0] # type:ignore
        except: customer_recency_days = 365
        try: customer_total_spend_ltm = df_stockcode.loc[df_stockcode['customerid'] == customerid, 'customer_total_spend_ltm_latest'].iloc[0] # type:ignore
        except: customer_total_spend_ltm = 0
        try: customer_freq_ltm =  df_stockcode.loc[df_stockcode['customerid'] == customerid, 'customer_freq_ltm_latest'].iloc[0] # type:ignore
        except: customer_freq_ltm = 0

        new_data = {
            'invoicedate': [np.datetime64(datetime.datetime.now())] * NUM_PRICE_BINS,
            'invoiceno': [data.get('invoiceno', np.nan)] * NUM_PRICE_BINS,
            'stockcode': [stockcode] * NUM_PRICE_BINS,
            'sales': [np.nan] * NUM_PRICE_BINS,
            'customerid': [customerid] * NUM_PRICE_BINS,
            'country': [data.get('country', df_stockcode.loc[0, 'country']) if df_stockcode is not None else np.nan] * NUM_PRICE_BINS,
            'unitprice': price_range,
            'product_avg_sales_last_month': [df_stockcode.loc[0, 'product_avg_sales_last_month'] if df_stockcode is not None else 0] * NUM_PRICE_BINS,
            'is_registered': [True if customerid else False] * NUM_PRICE_BINS,
            'customer_recency_days': [customer_recency_days] * NUM_PRICE_BINS,
            'customer_total_spend_ltm': [customer_total_spend_ltm] * NUM_PRICE_BINS,
            'customer_freq_ltm': [customer_freq_ltm] * NUM_PRICE_BINS,
            'is_return': [False] * NUM_PRICE_BINS,
        }
        new_df = pd.DataFrame(new_data)

        # add dt related features
        new_df['year'] = new_df['invoicedate'].dt.year
        new_df['year_month'] = new_df['invoicedate'].dt.to_period('M')
        new_df['day_of_week'] = new_df['invoicedate'].dt.strftime('%a')
        new_df['invoicedate'] = new_df['invoicedate'].astype(int) / 10 ** 9

        # suffle and transform input data
        target_col = 'sales'
        X = new_df.copy().drop(target_col, axis=1)
        X = X.sample(frac=1).reset_index(drop=True)
        if preprocessor: X = preprocessor.transform(X)

        # start prediction
        y_pred_actual = None
        epsilon = 1e-5
        if model:
            input_tensor = torch.tensor(X, dtype=torch.float32)
            model.eval()
            with torch.inference_mode():
                y_pred = model(input_tensor)
                y_pred = y_pred.cpu().numpy().flatten()
                y_pred_actual = np.exp(y_pred + epsilon)
                main_logger.info(f"primary model's prediction for stockcode {stockcode} - actual sales $ {y_pred_actual[0:5]}")

                if np.isinf(y_pred_actual).any() or (y_pred_actual == 0.0).any() or y_pred_actual is None: # type: ignore
                    if backup_model:
                        y_pred = backup_model.predict(X)
                        y_pred_actual = np.exp(y_pred + epsilon)
                        main_logger.info(f"backup model's prediction for stockcode {stockcode} - actual sales $ {y_pred_actual[0:5]}")

        elif backup_model:
            try: input_dim = len(backup_model.feature_name_)
            except: input_dim = len(backup_model.coef_)
            if X.shape[1] > input_dim: X = X[:, : X.shape[1] - input_dim]
            y_pred = backup_model.predict(X)
            y_pred_actual = np.exp(y_pred + epsilon)
            main_logger.info(f"backup model's prediction for stockcode {stockcode} -  actual sales $ {y_pred_actual[0:5]}")

        if y_pred_actual is not None:
            df_ = new_df.copy()
            df_['sales'] = y_pred_actual * 30
            df_ = df_.sort_values(by='unitprice')

            optimal_row = df_.loc[df_['sales'].idxmax()]
            optimal_price = optimal_row['unitprice']
            best_sales = optimal_row['sales']

            all_outputs = []
            for _, row in df_.iterrows():
                current_output = {
                    "stockcode": stockcode,
                    "unit_price": float(row['unitprice']),
                    "predicted_sales": float(row['sales']),
                    "optimal_unit_price": float(optimal_price), # type: ignore
                    "max_predicted_sales": float(best_sales), # type: ignore
                }
                all_outputs.append(current_output)

            main_logger.info(f'optimal price found: {optimal_price}')

            # store cached results and stockcode df
            if _redis_client is not None:
                if all_outputs[0]['optimal_unit_price'] != 0:
                    result_json = json.dumps(all_outputs)
                    _redis_client.set(cache_key_prediction_result_by_stockcode, result_json, ex=3600)

                if df_stockcode is not None:
                    df_stockcode_json = json.dumps(df_stockcode.to_dict())
                    _redis_client.set(cache_key_df_stockcode, df_stockcode_json, ex=86400)

                # deleted_df = _redis_client.delete(cache_key_df_stockcode)
                # deleted_prediction = _redis_client.delete(cache_key_prediction_result_by_stockcode)

            return jsonify(all_outputs)

        return jsonify([])

    except Exception as e:
        main_logger.error(f'error: {e}')
        return jsonify([])


# sage maker's standard endpoint
@app.route('/ping', methods=['GET'])
def ping():
    return '', 200


@app.route('/invocations', methods=['POST'])
def invocations():
    try:
        data = request.get_json(force=True)
        stockcode = data.get('stock_code', '85123A')
    except Exception as e:
        return jsonify({"error": f"Error parsing JSON: {str(e)}"}), 400

    result = predict_price(stockcode)
    return jsonify(result), 200


# add headers (cors handling for lambda)
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000' if 'localhost' in request.host else CLIENT_A
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONSS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response


# loading
main_logger.info("=== START LOADING ===")

# preprocessor
start_time = time.time()
load_preprocessor()
main_logger.info(f"preprocessor took: {time.time() - start_time:.2f} seconds")

# models
start_time = time.time()
try: load_artifacts_primary_model()
except: load_artifacts_backup_model()
main_logger.info(f"model loading took: {time.time() - start_time:.2f} seconds")



def handle_function_url_request(event, context):
    try:
        # try bypass api gateway
        path = event['requestContext']['http']['path']
        method = event['requestContext']['http']['method']
        if method == 'POST' and '/predict-price/' in path:
            # extract stockcode from path
            stockcode = path.split('/predict-price/')[-1]
            result = predict_price(stockcode)
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({ **result })
            }

        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Not found'})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def handler(event, context):
    main_logger.info("... lambda handler invoked.")

    # parse function URL event, bypassing api gateway
    if 'requestContext' in event and 'http' in event['requestContext']:
        return handle_function_url_request(event, context)

    # create redis client
    main_logger.info("=== START REDIS CLIENT ===")
    start_time = time.time()
    try: get_redis_client()
    except Exception as e: main_logger.warning(f"failed to establish initial Redis connection in handler: {e}")

    if _redis_client is not None:
        # test connection
        _redis_client.ping()
        main_logger.info(f"✅ redis ping successful: {time.time() - start_time:.2f} seconds")

        # test a simple operation
        start_time = time.time()
        _redis_client.set("test_key", "test_value", ex=60)
        main_logger.info(f"✅ redis set operation: {time.time() - start_time:.2f} seconds")

        # _redis_client.flushall(target_nodes='all')
        all_keys = _redis_client.keys('*')
        if all_keys:
            deleted = _redis_client.delete(*all_keys)
            main_logger.info(f"✅ manually deleted {deleted} keys")

    return awsgi.response(app, event, context)


if __name__ == '__main__':
    if os.environ.get('ENV') == 'local':
        main_logger.info("... running flask app with waitress for local development ... ")
        serve(app, host='0.0.0.0', port=5002)
    else:
        app.run(host='0.0.0.0', port=8080)
