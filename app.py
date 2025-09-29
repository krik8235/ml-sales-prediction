import os
import boto3
import json
import time
import math
import warnings
import hashlib
import torch
import pickle
import pandas as pd
import numpy as np
import awsgi
import joblib
import redis
import redis.cluster
from redis.cluster import ClusterNode
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from waitress import serve
from dotenv import load_dotenv
import dvc.api
from dvc.exceptions import DvcException

import src.model.torch_model as t
import src.data_handling as data_handling
from src._utils import main_logger, s3_extract

# silence warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# load env
AWS_LAMBDA_RUNTIME_API = os.environ.get('AWS_LAMBDA_RUNTIME_API', None)
if AWS_LAMBDA_RUNTIME_API is None: load_dotenv(override=True)

ENV = os.environ.get('ENV', 'production')


# file path
DFN_FILE_PATH = os.path.join('models', 'production', 'dfn_best.pth')
GBM_FILE_PATH =  os.path.join('models', 'production', 'gbm_best.pth')
SVR_FILE_PATH = os.path.join('models', 'production', 'svr_best.pth')
EN_FILE_PATH = os.path.join('models', 'production', 'en_best.pth')

PREPROCESSOR_PATH = os.path.join('preprocessors', 'column_transformer.pkl')
ORIGINAL_DF_PATH = os.environ.get('ORIGINAL_DF_PATH')
X_TEST_PATH = os.environ.get('X_TEST', 'data/x_test_df.parquet')


# s3 boto client
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-sales-pred')
s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'))

try:
    # test boto3 client
    sts_client = boto3.client('sts')
    identity = sts_client.get_caller_identity()
    main_logger.info(f"✅ Boto3 is using a role: {identity['Arn']}")

    # test s3 access
    response = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=PREPROCESSOR_PATH)
    main_logger.info("✅ S3 access works")

except Exception as e:
    main_logger.error(f"❌ Boto3 / S3 credentials/permissions error: {e}")


# flask app config
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CLIENT_A =  os.environ.get('CLIENT_A')
API_ENDPOINT = os.environ.get('API_ENDPOINT')
origins = ['http://localhost:3000', CLIENT_A, API_ENDPOINT]


# global variables
_redis_client = None
X_test = None
preprocessor = None
model = None
backup_model = None

device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device_type)


# redis client
def get_redis_client():
    global _redis_client

    if _redis_client is None:
        REDIS_HOST = os.environ.get('REDIS_HOST')
        REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
        REDIS_TLS = os.environ.get('REDIS_TLS', 'true').lower() == 'true'

        if not REDIS_HOST:
            main_logger.error("REDIS_HOST environment variable not set. Redis connection will fail.")

        try:
            startup_nodes = [ClusterNode(host=REDIS_HOST, port=REDIS_PORT)]
            _redis_client = redis.cluster.RedisCluster(
                startup_nodes=startup_nodes,
                decode_responses=True,
                skip_full_coverage_check=True,
                ssl=REDIS_TLS,                  # elasticache has encryption in transit: enabled -> must be true
                ssl_cert_reqs=None,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
                retry_on_timeout=True,
                retry_on_error=[
                    redis.ConnectionError,
                    redis.TimeoutError
                ],
                max_connections=10,            # limit connections for Lambda
                max_connections_per_node=2     # limit per node
            )
            _redis_client.ping()
            main_logger.info("successfully connected to ElastiCache Redis Cluster (Configuration Endpoint)")
        except redis.ConnectionError as e:
            main_logger.error(f"could not connect to Redis Cluster: {e}", exc_info=True)
            _redis_client = None
        except Exception as e:
            main_logger.error(f"an unexpected error occurred during Redis Cluster connection: {e}", exc_info=True)
            _redis_client = None
    return _redis_client


# dvc config
DVC_REMOTE_NAME = 'storage'
DVC_REMOTE_URL = os.environ.get('DVC_REMOTE_URL_ENV', 's3://ml-sales-pred/dvc')
DVC_TMP_DIR = '/tmp'


def configure_dvc_for_lambda():
    # set dvc directories to /tmp
    os.environ.update({
        'DVC_CACHE_DIR': '/tmp/dvc-cache',
        'DVC_DATA_DIR': '/tmp/dvc-data',
        'DVC_CONFIG_DIR': '/tmp/dvc-config',
        'DVC_GLOBAL_CONFIG_DIR': '/tmp/dvc-global-config',
        'DVC_SITE_CACHE_DIR': '/tmp/dvc-site-cache'
    })
    for dir_path in ['/tmp/dvc-cache', '/tmp/dvc-data', '/tmp/dvc-config']:
        os.makedirs(dir_path, exist_ok=True)


def load_x_test():
    global X_test
    if not os.environ.get('PYTEST_RUN', False):
        main_logger.info("... loading x_test ...")

        try:
            with dvc.api.open(X_TEST_PATH, remote=DVC_REMOTE_NAME, mode='rb') as fd:
                X_test = pd.read_parquet(fd)
                main_logger.info('✅ successfully loaded x_test via dvc api')

        except DvcException as e:
            main_logger.error(f'❌ dvc error: failed to load from dvc remote: {e}', exc_info=True)

        except Exception as e:
            main_logger.error(f'❌ general loading error: {e}', exc_info=True)


def load_preprocessor():
    global preprocessor
    if not os.environ.get('PYTEST_RUN', False):
        main_logger.info("... loading preprocessor ...")
        configure_dvc_for_lambda()
        try:
            with dvc.api.open(PREPROCESSOR_PATH, remote=DVC_REMOTE_NAME, mode='rb') as fd:
                preprocessor = joblib.load(fd)
                main_logger.info('✅ successfully loaded preprocessor via dvc api')

        except DvcException as e:
            main_logger.error(f'❌ dvc api error: failed from dvc remote: {e}', exc_info=True)

        except Exception as e:
            main_logger.error(f'❌ general loading error: {e}', exc_info=True)


def load_model(stockcode: str = ''):
    global model, backup_model

    if stockcode:
        DFN_FILE_PATH_STOCKCODE = os.path.join('models', 'production', f'dfn_best_{stockcode}.pth')
        try:
            main_logger.info('... loading artifacts - trained dfn by stockcode ...')
            with dvc.api.open(DFN_FILE_PATH_STOCKCODE, repo=os.getcwd(), mode='rb') as fd:
                checkpoint = torch.load(fd, weights_only=False, map_location=device)
                main_logger.info('✅ successfully loaded dfn via dvc api')
            model = t.scripts.load_model(checkpoint=checkpoint, file_path=DFN_FILE_PATH_STOCKCODE)
            model.eval()
            main_logger.info(f'loaded trained dfn by stockcode: {stockcode}')

        except:
            try:
                model_data_bytes_io = s3_extract(file_path=DFN_FILE_PATH_STOCKCODE)
                checkpoint = torch.load(model_data_bytes_io, weights_only=False, map_location=device) # type: ignore
                model = t.scripts.load_model(checkpoint=checkpoint, file_path=DFN_FILE_PATH_STOCKCODE)
                model.eval()
                main_logger.info(f'loaded trained dfn by stockcode: {stockcode}')
            except:
                try:
                    main_logger.info('... loading artifacts - trained dfn full ...')
                    model_data_bytes_io_ = s3_extract(file_path=DFN_FILE_PATH)
                    checkpoint_ = torch.load(model_data_bytes_io_, weights_only=False, map_location=device) # type: ignore
                    model = t.scripts.load_model(checkpoint=checkpoint_, file_path=DFN_FILE_PATH)
                    model.eval()
                    main_logger.info('loaded trained dfn overall')
                except:
                    load_artifacts_backup_model()
    else:
        try:
            main_logger.info('... loading artifacts - trained dfn ...')
            model_data_bytes_io_ = s3_extract(file_path=DFN_FILE_PATH)
            checkpoint_ = torch.load(model_data_bytes_io_, weights_only=False, map_location=device) # type: ignore
            model = t.scripts.load_model(checkpoint=checkpoint_, file_path=DFN_FILE_PATH)
            model.eval()
            main_logger.info('loaded trained dfn overall')
        except:
            load_artifacts_backup_model()


def load_artifacts_primary_model():
    global model

    try:
        # load trained processor
        main_logger.info("... loading artifacts - trained dfn ...")
        model_data_bytes_io = s3_extract(file_path=DFN_FILE_PATH)

        if model_data_bytes_io is not None:
            checkpoint = torch.load(model_data_bytes_io, weights_only=False, map_location=device)
            model = t.scripts.load_model(checkpoint=checkpoint, trig='best')
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
        model_data_bytes_io = s3_extract(file_path=GBM_FILE_PATH)
        if model_data_bytes_io:
            model_data_bytes_io.seek(0)
            loaded_dict = pickle.load(model_data_bytes_io)
            backup_model = loaded_dict['best_model']
            main_logger.info("... successfully loaded gbm ...")
            return

    except:
        main_logger.error(f"failed to load gbm from s3 using pickle. try loading svr instead as a backup model.")
        try:
            main_logger.info('... loading svr ...')
            model_data_bytes_io = s3_extract(file_path=SVR_FILE_PATH)
            if model_data_bytes_io:
                model_data_bytes_io.seek(0)
                loaded_dict = pickle.load(model_data_bytes_io)
                backup_model = loaded_dict['best_model']
                main_logger.info("... successfully loaded svr ...")
                return
        except Exception as e:
            main_logger.critical(f"failed to load svr from s3 using pickle. try loading elastic net instead as a backup model.")
            try:
                main_logger.info('... loading en ...')
                model_data_bytes_io = s3_extract(file_path=EN_FILE_PATH)
                if model_data_bytes_io:
                    model_data_bytes_io.seek(0)
                    loaded_dict = pickle.load(model_data_bytes_io)
                    backup_model = loaded_dict['best_model']
                    main_logger.info("... successfully loaded elastic net ...")
                    return
            except Exception as e:
                    main_logger.critical(f"failed to load backup model: {e}")



# api endpoints
@app.route('/')
def hello_world():
    if os.environ.get('ENV') == 'local': return """<p>Hello, world</p><p>I am an API endpoint.</p>"""

    data = request.json if request.is_json else request.data.decode('utf-8')
    main_logger.info(f"request received! ENV: {ENV}, Data: {data}")
    return jsonify({"message": f"Hello from Flask in Lambda! ENV: {ENV}", "received_data": data})


@app.route('/v1/predict-price/<string:stockcode>', methods=['GET', 'OPTIONS'])
@cross_origin(origins=origins, methods=['GET', 'OPTIONS'], supports_credentials=True)
def predict_price(stockcode):
    df_stockcode = None

    # try:
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
            try: return jsonify(json.loads(cached_prediction_result)) # type: ignore
            except: return jsonify(json.loads(json.dumps(cached_prediction_result)))

        # load historical data of the product (stockcode)
        cached_df_stockcode = _redis_client.get(cache_key_df_stockcode)
        if cached_df_stockcode: df_stockcode = json.loads(json.dumps(cached_df_stockcode))

    if df_stockcode is None:
        df_stockcode_bites_io = s3_extract(file_path=file_path_df_stockcode[1:] if file_path_df_stockcode[0] == '/' else file_path_df_stockcode)
        if df_stockcode_bites_io: df_stockcode = pd.read_parquet(df_stockcode_bites_io)

    # define the price range
    min_price = float(data.get('unitprice_min', 0.0))
    max_price = float(data.get('unitprice_max', 0.0))

    if min_price == 0.0:
        if df_stockcode is not None and not df_stockcode.empty:
            min_price = df_stockcode['unitprice_min'][0]
            if min_price < 0.1: min_price = df_stockcode['unitprice_median'][0] # type: ignore
        else:
            min_price = 2.0

    if max_price == 0.0:
        if df_stockcode is not None and not df_stockcode.empty:
            max_price = df_stockcode['unitprice_max'][0]
        else:
            max_price = 20.0

    if min_price == max_price:
        min_price = max(2, min_price * 0.9)
        max_price = max_price * 1.5 + 0.1
    elif min_price > max_price:
        min_price, max_price = max_price, min_price

    if min_price > 1:
        min_price = math.floor(min_price)

    if not 'unitprice_max' in data and max_price - min_price < 10.0:
        max_price = max_price * 5


    NUM_PRICE_BINS = int(data.get('num_price_bins', 100))
    price_range = np.linspace(min_price, max_price, NUM_PRICE_BINS)
    main_logger.info(f'price range for stockcode {stockcode}: ${min_price} - ${max_price}')

    if X_test is not None:
        # create df
        price_range_df = pd.DataFrame({ 'unitprice': price_range })
        test_sample = X_test.sample(n=1200, random_state=42, ignore_index=True) # type: ignore
        test_sample_merged = test_sample.merge(price_range_df, how='cross') if X_test is not None else price_range_df
        test_sample_merged.drop('unitprice_x', axis=1, inplace=True)
        test_sample_merged.rename(columns={'unitprice_y': 'unitprice'}, inplace=True)

        X = preprocessor.transform(test_sample_merged) if preprocessor else test_sample_merged

        # load model
        try: load_model(stockcode=stockcode)
        except: pass

        # perform inference
        y_pred_actual = None
        epsilon = 0
        if model:
            input_tensor = torch.tensor(X, dtype=torch.float32)
            model.eval()
            with torch.inference_mode():
                y_pred = model(input_tensor)
                y_pred = y_pred.cpu().numpy().flatten()
                y_pred_actual = np.exp(y_pred + epsilon)
                main_logger.info(f"primary model's prediction for stockcode {stockcode} - actual quantity {y_pred_actual[0:5]}")

                if np.isinf(y_pred_actual).any() or (y_pred_actual == 0.0).any() or y_pred_actual is None: # type: ignore
                    if backup_model:
                        y_pred = backup_model.predict(X)
                        y_pred_actual = np.exp(y_pred + epsilon)
                        main_logger.info(f"backup model's prediction for stockcode {stockcode} - actual quantity {y_pred_actual[0:5]}")

        elif backup_model:
            try: input_dim = len(backup_model.feature_name_)
            except: input_dim = len(backup_model.coef_)
            if X.shape[1] > input_dim: X = X[:, : X.shape[1] - input_dim]
            y_pred = backup_model.predict(X)
            y_pred_actual = np.exp(y_pred + epsilon)
            main_logger.info(f"backup model's prediction for stockcode {stockcode} -  actual quantity {y_pred_actual[0:5]}")

        if y_pred_actual is not None:
            df_ = test_sample_merged.copy()
            df_['quantity'] = np.floor(y_pred_actual * 100)
            df_['sales'] = df_['quantity'] * df_['unitprice']
            df_ = df_.sort_values(by='unitprice')

            df_results = df_.groupby('unitprice').agg(
                quantity=('quantity', 'mean'),
                quantity_min=('quantity', 'min'),
                quantity_max=('quantity', 'max'),
                sales=('sales', 'mean'),
            ).reset_index()

            optimal_row = df_results.loc[df_results['sales'].idxmax()]
            optimal_price = optimal_row['unitprice']
            optimal_quantity = optimal_row['quantity']
            best_sales = optimal_row['sales']

            all_outputs = []
            for _, row in df_results.iterrows():
                current_output = {
                    "stockcode": stockcode,
                    "unit_price": float(row['unitprice']),
                    'quantity': int(row['quantity']),
                    'quantity_min': int(row['quantity_min']),
                    'quantity_max': int(row['quantity_max']),
                    "predicted_sales": float(row['sales']),
                    "optimal_unit_price": float(optimal_price), # type: ignore
                    "max_predicted_sales": float(best_sales), # type: ignore
                }
                all_outputs.append(current_output)

            # store the prediction results in cache
            if all_outputs and _redis_client is not None:
                serialized_data = json.dumps(all_outputs)
                _redis_client.set(cache_key_prediction_result_by_stockcode, serialized_data, ex=3600) # expire in an hour

            main_logger.info(f'optimal price: $ {optimal_price:,.2f}, quantity: {optimal_quantity:,.0f}, maximum sales: $ {best_sales:,.2f}')
            return jsonify(all_outputs)

        return jsonify([])

    # except Exception as e:
    #     main_logger.error(f'error: {e}')
    #     return jsonify([])


# sagemaker's standard endpoint
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
main_logger.info(f"preprocessor loading took: {time.time() - start_time:.2f} seconds")

start_time = time.time()
load_x_test()
main_logger.info(f"x test loading took: {time.time() - start_time:.2f} seconds")



def handle_function_url_request(event, context):
    try:
        # try bypassing api gateway
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

        _redis_client.flushall(target_nodes='all')
        all_keys = _redis_client.keys('*')
        if all_keys:
            deleted = _redis_client.delete(*all_keys) # type: ignore
            main_logger.info(f"✅ manually deleted {deleted} keys")

    return awsgi.response(app, event, context)


if __name__ == '__main__':
    if ENV == 'local':
        main_logger.info("... running flask app with waitress for local development ... ")
        serve(app, host='0.0.0.0', port=5002)
    else:
        app.run(host='0.0.0.0', port=8080)
