import os
import json
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
from flask import Flask, request, jsonify # type: ignore
from flask_cors import cross_origin
from waitress import serve
from dotenv import load_dotenv # type: ignore

import src.model.torch_model as t
import src.data_handling as data_handling
from src._utils import main_logger, s3_load, S3_BUCKET_NAME, s3_upload

# silence warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# load env
AWS_LAMBDA_RUNTIME_API = os.environ.get('AWS_LAMBDA_RUNTIME_API', None)
if AWS_LAMBDA_RUNTIME_API is None: load_dotenv(override=True)


# flask app config
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CLIENT_A =  os.environ.get('CLIENT_A')
API_ENDPOINT = os.environ.get('API_ENDPOINT')
origins = ['http://localhost:3000', CLIENT_A, API_ENDPOINT]


# global variables
_redis_client = None
original_df = None
preprocessor = None
model = None
backup_model = None


# file path
PRODUCTION_MODEL_FOLDER_PATH = 'models/production'
DFN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'dfn_best.pth')
GBM_FILE_PATH =  os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'gbm_best.pth')
EN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'en_best.pth')

PREPROCESSOR_PATH = 'preprocessors/column_transformer.pkl'
ORIGINAL_DF_PATH = os.environ.get('ORIGINAL_DF_PATH')


def get_redis_client():
    global _redis_client

    if _redis_client is None:
        REDIS_HOST = os.environ.get("REDIS_HOST", None) 
        REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
        REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
        REDIS_TLS = os.environ.get("REDIS_TLS", "false").lower() == "true"

        if not REDIS_HOST:
            main_logger.error("REDIS_HOST environment variable not set. Redis connection will fail.")
            raise ValueError("REDIS_HOST environment variable is required.")

        try:
            startup_nodes = [ClusterNode(host=REDIS_HOST, port=REDIS_PORT)]
            _redis_client = redis.cluster.RedisCluster(
                startup_nodes=startup_nodes,
                password=REDIS_PASSWORD,
                decode_responses=True,
                skip_full_coverage_check=True,
                ssl=REDIS_TLS,
                ssl_cert_reqs=None,
            )
            _redis_client.ping()
            main_logger.info("successfully connected to ElastiCache Redis Cluster (Configuration Endpoint)!")
        except redis.exceptions.ConnectionError as e:
            main_logger.error(f"could not connect to Redis Cluster: {e}", exc_info=True)
            _redis_client = None
            raise
        except Exception as e:
            main_logger.error(f"an unexpected error occurred during Redis Cluster connection: {e}", exc_info=True)
            _redis_client = None
            raise
    return _redis_client


def load_artifacts():
    global model, preprocessor, original_df

    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)

    try:
        # load the original dataframe
        main_logger.info("... loading original dataframe ...")
        original_df = data_handling.scripts.load_original_dataframe()
        
        # load trained processor
        main_logger.info("... loading pre-processing artifacts...")
        try:
            preprocessor_file_path = s3_load(PREPROCESSOR_PATH)
            preprocessor = joblib.load(preprocessor_file_path)
        except:
            preprocessor = joblib.load(PREPROCESSOR_PATH)

        # load pytorch model
        main_logger.info("... loading pytorch model...")
        input_dim = 59
        model_io_file = s3_load(file_path=DFN_FILE_PATH)

        if model_io_file is not None:
            state_dict = torch.load(model_io_file, weights_only=False, map_location=device)
            model = t.scripts.load_model(input_dim=input_dim, trig='best', saved_state_dict=state_dict)
            model.eval()
        else:
            model = t.scripts.load_model(input_dim=input_dim, trig='best')
            model.eval()

    except Exception as e:
        main_logger.critical(f"failed to load one or more essential artifacts during cold start: {e}", exc_info=True)
        raise RuntimeError("flask application failed to initialize due to missing or invalid model artifacts.")


def load_backup_model():
    global backup_model
    model_data_bytes_io = None
    
    try:
        main_logger.info('... loading gbm ...')
        model_data_bytes_io = s3_load(file_path=GBM_FILE_PATH)

        if model_data_bytes_io:
            model_data_bytes_io.seek(0)
            loaded_dict = pickle.load(model_data_bytes_io)
            backup_model = loaded_dict['best_model']
            main_logger.info("successfully loaded gbm.")
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
                main_logger.info("successfully loaded elastic net.")
                return

        except Exception as e:
            main_logger.critical(f"failed to load elastic net from s3 using pickle: {e}")

try:
    load_artifacts()
except:
    load_backup_model()


@app.route('/')
def hello_world():
    env = os.environ.get('ENV', None)
    if env == 'local': return """<p>Hello, world</p><p>I am an API endpoint.</p>"""

    data = request.json if request.is_json else request.data.decode('utf-8')
    main_logger.info(f"request received! ENV: {env}, Data: {data}")
    return jsonify({"message": f"Hello from Flask in Lambda! ENV: {env}", "received_data": data})



@app.route('/v1/predict-price/<string:stockcode>', methods=['GET', 'OPTION'])
@cross_origin(origins=origins, methods=['GET', 'OPTION'], supports_credentials=True)
def predict_price(stockcode):
    data = request.json if request.is_json else {}

    # fetch cached results if any
    if _redis_client is not None:
        params_hash = hashlib.sha256(str(sorted(data.items())).encode()).hexdigest()
        cache_key = f"prediction:{stockcode}:{params_hash}"
        cached_result = _redis_client.get(cache_key)
        if cached_result:
            main_logger.info(f"cache hit for stockcode: {stockcode}")
            return jsonify(json.loads(cached_result))
    

    NUM_PRICE_BINS = data.get('num_price_bins', 12) if data else 12

    stockcode = stockcode if stockcode else data.get('stockcode', '85123A') if data is not None else '85123A'
    df_target_stockcode = original_df.copy()[original_df['stockcode'] == stockcode] # type: ignore

    if df_target_stockcode.empty:
        min_price = 2
        max_price = 20.0
        main_logger.warning(f"no historical data found for stockcode: {stockcode}. use default price range.")
    else:
        min_price = max(df_target_stockcode['unitprice'].min(), 2)
        max_price = df_target_stockcode['unitprice'].max()
    if min_price == max_price:
        min_price = max(2, min_price * 0.9)
        max_price = max_price * 1.1 + 0.1
    elif min_price > max_price:
        min_price, max_price = max_price, min_price

    price_range = np.linspace(min_price, max_price, NUM_PRICE_BINS)

    # mean_quantity = df_target_stockcode['quantity'].median()
    country = df_target_stockcode['country'].mode().iloc[0]   
    new_data = {
        'invoicedate': np.datetime64(datetime.datetime.now()),
        'invoiceno': [data.get('invoiceno', np.nan) if data else np.nan] * NUM_PRICE_BINS,
        'stockcode': [stockcode] * NUM_PRICE_BINS,
        'description': [np.nan] * NUM_PRICE_BINS,
        # 'quantity': [np.int64(data.get('quantity', np.nan) if data else np.nan)] * NUM_PRICE_BINS,
        'customerid': [np.nan] * NUM_PRICE_BINS,
        'country': [data.get('country', country) if data else country] * NUM_PRICE_BINS,
        'unitprice': price_range
    }
    new_df = pd.DataFrame(new_data)
    df = pd.concat([
        # original_df, 
        new_df], ignore_index=True)
    df = data_handling.scripts.structure_missing_values(df=df)
    df = data_handling.scripts.handle_feature_engineering(df=df)

    target_col = 'quantity'
    X = df.copy().drop(target_col, axis=1)
    X = X.tail(NUM_PRICE_BINS)

    if preprocessor:
        X = preprocessor.transform(X)

    y_pred_actual = None
    if model:
        input_tensor = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.inference_mode():
            y_pred = model(input_tensor)
            y_pred = y_pred.cpu().numpy().flatten()
            y_pred_actual = np.exp(y_pred)
            main_logger.info(f"primary model's prediction for stockcode {stockcode} -  actual quantity (units) {y_pred_actual}")

            if np.isinf(y_pred_actual).any() or (y_pred_actual == 0.0).any() or y_pred_actual is None: # type: ignore   
                if backup_model:
                    y_pred = backup_model.predict(X)
                    y_pred_actual = np.exp(y_pred)
                    main_logger.info(f"backup model's prediction for stockcode {stockcode} -  actual quantity (units) {y_pred_actual}")
        
    else:
        if backup_model:
            y_pred = backup_model.predict(X)
            y_pred_actual = np.exp(y_pred)
            main_logger.info(f"backup model's prediction for stockcode {stockcode} -  actual quantity (units) {y_pred_actual}")
    

    if y_pred_actual is not None:
        new_df['sales'] =  y_pred_actual
        optimal_row = new_df.loc[new_df['sales'].idxmax()]
        optimal_price = optimal_row['unitprice']
        best_sales = optimal_row['sales']

        all_outputs = []
        for _, row in new_df.iterrows():
            current_output = {
                "stockcode": stockcode,
                "unit_price": float(row['unitprice']),
                "predicted_sales": float(row['sales']),
                "optimal_unit_price": float(optimal_price), # type: ignore
                "max_predicted_sales": float(best_sales), # type: ignore
            }
            all_outputs.append(current_output)

        # store cached results
        if _redis_client is not None:
            result_json = json.dumps(all_outputs)
            _redis_client.set(cache_key, result_json, ex=3600)

        return jsonify(all_outputs)
    
    else:
        return jsonify([])


@app.after_request
def add_header(response):   
    response.headers['Cache-Control'] = 'public, max-age=0'
    response.headers['Access-Control-Allow-Origin'] = CLIENT_A
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response


def handler(event, context):
    main_logger.info("lambda handler invoked.")

    try:
        get_redis_client()
    except Exception as e:
        main_logger.critical(f"failed to establish initial Redis connection in handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to initialize Redis client. Check environment variables and network config.'})
        }
    
    return awsgi.response(app, event, context)


if __name__ == '__main__':
    if os.environ.get('ENV') == 'local':
        main_logger.info("...running flask app with waitress for local development...")
        serve(app, host='0.0.0.0', port=5002)

    else:
        app.run(host='0.0.0.0', port=5002)