import os
import datetime
import warnings
import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv # type: ignore
import joblib
import redis # type: ignore
import redis.cluster # type: ignore
from redis.cluster import ClusterNode # type: ignore
from flask import Flask, request, jsonify # type: ignore
from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS, cross_origin
from waitress import serve
import awsgi # type: ignore

import src.model.torch_model.scripts as t
import src.model.sklearn_model.scripts as sk
import src.data_handling as data_handling
from src._utils import main_logger

# silence warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# load env
AWS_LAMBDA_RUNTIME_API = os.environ.get('AWS_LAMBDA_RUNTIME_API', None)
if AWS_LAMBDA_RUNTIME_API is None: load_dotenv(override=True)


# flask app config
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
origins = ['http://localhost:3000', os.environ.get('MY_WEB'), os.environ.get('PRODUCTION_API_ROUTE') ]
cors = CORS(app, resources={r'/v1/*': { 'origins': origins, }})


# global variables
_redis_client = None
model = None
preprocessor = None
backup_model = None
original_df = None
preprocessed_df = None


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
            main_logger.info("Successfully connected to ElastiCache Redis Cluster (Configuration Endpoint)!")
        except redis.exceptions.ConnectionError as e:
            main_logger.error(f"Could not connect to Redis Cluster: {e}", exc_info=True)
            _redis_client = None
            raise
        except Exception as e:
            main_logger.error(f"An unexpected error occurred during Redis Cluster connection: {e}", exc_info=True)
            _redis_client = None
            raise
    return _redis_client


def load_artifacts():
    global model, preprocessor, original_df, preprocessed_df

    # load df
    main_logger.info('loading original dataframe...')
    original_df = data_handling.scripts.load_original_dataframe()
    preprocessed_df, _, _ = data_handling.scripts.load_post_feature_engineer_dataframe()
    preprocessed_df = preprocessed_df.drop('Unnamed: 0', axis=1)

    # load the trained PyTorch model
    main_logger.info("loading model...")
    input_dim = 59
    model = t.load_model(input_dim=input_dim, trig='grid')
    model.eval()

    main_logger.info("loading pre-processing artifacts...")
    PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', None)
    target_col = 'sales'
    if PREPROCESSOR_PATH and os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    else:
        num_cols, cat_cols = data_handling.scripts.categorize_num_cat_cols(df=preprocessed_df, target_col=target_col)
        preprocessor = data_handling.scripts.handle_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
        
    X_train = preprocessed_df.copy().drop(target_col, axis=1)
    preprocessor.fit(X_train)


def load_backup_model():
    global backup_model
    try:
        backup_model, _ = sk.load_model(model_name='gbm', trig='best')
    except:
        try:
            backup_model, _ = sk.load_model(model_name='en', trig='best')
        except:
            backup_model = None


load_artifacts()

load_backup_model()

# backup_model = ElasticNet(
#     alpha=0.00010076137476187376,
#     l1_ratio=0.6001201735703293,
#     max_iter=48953,
#     tol=1.8200116325906476e-05,
#     selection='random',
#     fit_intercept=True,
#     random_state=96
# )


@app.route('/')
def hello_world():
    env = os.environ.get('ENV', None)
    if env == 'local': return """<p>Hello, world</p><p>I am an API endpoint.</p>"""

    data = request.json if request.is_json else request.data.decode('utf-8')
    main_logger.info(f"Request received! ENV: {env}, Data: {data}")
    return jsonify({"message": f"Hello from Flask in Lambda! ENV: {env}", "received_data": data})


@app.route('/v1/predict-price/<string:stockcode>', methods=['GET'])
@cross_origin(origins=origins)
def predict_price(stockcode):
    # create new dataframe
    try: data = request.json
    except: data = None

    NUM_PRICE_BINS = data.get('num_price_bins', 12) if data else 12

    stockcode = stockcode if stockcode else data.get('stockcode', '85123A') if data is not None else '85123A'
    df_target_stockcode = original_df.copy()[original_df['stockcode'] == stockcode] # type: ignore

    if df_target_stockcode.empty:
        min_price = 2
        max_price = 20.0
        main_logger.warning(f"No historical data found for stockcode: {stockcode}. Using default price range.")
    else:
        min_price = max(df_target_stockcode['unitprice'].min(), 2)
        max_price = df_target_stockcode['unitprice'].max()
    if min_price == max_price:
        min_price = max(2, min_price * 0.9)
        max_price = max_price * 1.1 + 0.1
    elif min_price > max_price:
        min_price, max_price = max_price, min_price

    price_range = np.linspace(min_price, max_price, NUM_PRICE_BINS)
    mean_quantity = df_target_stockcode['quantity'].median()
    country = df_target_stockcode['country'].mode().iloc[0]   
    new_data = {
        'invoicedate': np.datetime64(datetime.datetime.now()),
        'invoiceno': [data.get('invoiceno', np.nan) if data else np.nan] * NUM_PRICE_BINS,
        'stockcode': [stockcode] * NUM_PRICE_BINS,
        'description': [np.nan] * NUM_PRICE_BINS,
        'quantity': [np.int64(data.get('quantity', mean_quantity) if data else mean_quantity)] * NUM_PRICE_BINS,
        'customerid': [np.nan] * NUM_PRICE_BINS,
        'country': [data.get('country', country) if data else country] * NUM_PRICE_BINS,
        'unitprice': price_range
    }
    new_df = pd.DataFrame(new_data)
    df = pd.concat([original_df, new_df], ignore_index=True)
    df = data_handling.scripts.structure_missing_values(df=df)
    df = data_handling.scripts.handle_feature_engineering(df=df)

    target_col = 'sales'  
    X = df.copy().drop(target_col, axis=1)
    X = X.tail(NUM_PRICE_BINS)

    if preprocessor:
        X = preprocessor.transform(X)

    y_pred_actual = None
    if model:
        input_tensor = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            y_pred = model(input_tensor)
            y_pred = y_pred.cpu().numpy().flatten()
            y_pred_actual = np.exp(y_pred)

            main_logger.info(f"\nPrediction for stockcode {stockcode} - Atual Sales Predicted in $: {y_pred_actual}")

    if np.isinf(y_pred_actual).any() or (y_pred_actual == 0.0).any() or y_pred_actual is None: # type: ignore   
        if backup_model:
            y_pred = backup_model.predict(X)
            y_pred_actual = np.exp(y_pred)
    
    elif not model and backup_model:
        y_pred = backup_model.predict(X)
        y_pred_actual = np.exp(y_pred)
      
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

        return jsonify(all_outputs)
    
    else:
        return jsonify([])

        # yield json.dumps(current_output) + '\n'
    # return Response(generate_predictions_stream(), mimetype='application/json') 

@app.after_request
def add_header(response):   
    # response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


def handler(event, context):
    import json

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