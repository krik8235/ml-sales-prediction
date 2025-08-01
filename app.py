import os
import json
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
from flask import Flask, request, jsonify, Response # type: ignore
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


# global variables
_redis_client = None
model = None
preprocessor = None
backup_model = None


# flask app config
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
origins = ['http://localhost:3000', os.environ.get('MY_WEB'), os.environ.get('PRODUCTION_API_ROUTE') ]
cors = CORS(app, resources={r'/v1/*': { 'origins': origins, }})

@app.route('/')
def hello_world():
    env = os.environ.get('ENV', None)
    if env == 'local': return """<p>Hello, world</p><p>I am an API endpoint.</p>"""

    data = request.json if request.is_json else request.data.decode('utf-8')
    main_logger.info(f"Request received! ENV: {env}, Data: {data}")
    return jsonify({"message": f"Hello from Flask in Lambda! ENV: {env}", "received_data": data})
   


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
    global model, preprocessor, input_dim

    PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', None)
    input_dim = 59

    if PREPROCESSOR_PATH is None or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
   
    # Load the trained PyTorch model
    main_logger.info("Loading model...")
    model = t.load_model(input_dim=input_dim, trig='grid')
    model.eval()

    main_logger.info("Loading pre-processing artifacts...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)



def load_backup_model():
    global backup_model
    try:
        backup_model, _ = sk.load_model(model_name='gbm', trig='best')
    except:
        try:
            backup_model, _ = sk.load_model(model_name='en', trig='best')
        except:
            backup_model = None


# backup_model = ElasticNet(
#     alpha=0.00010076137476187376,
#     l1_ratio=0.6001201735703293,
#     max_iter=48953,
#     tol=1.8200116325906476e-05,
#     selection='random',
#     fit_intercept=True,
#     random_state=96
# )

@app.route('/v1/predict-price/<string:stockcode>', methods=['GET'])
@cross_origin(origins=origins)
def predict_price(stockcode):
    # load original dataframe
    original_df = data_handling.scripts.load_original_dataframe()

    # create new dataframe
    try: data = request.json
    except: data = None

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

    price_range = np.linspace(min_price, max_price, 10)

    mean_quantity = df_target_stockcode['quantity'].median()


    new_data = {
        'invoicedate': np.datetime64(datetime.datetime.now()),
        'invoiceno': data.get('invoiceno', np.nan) if data else np.nan,
        'stockcode': stockcode,
        'description': np.nan,
        'quantity': np.int64(data.get('quantity', mean_quantity) if data else mean_quantity),
        'customerid': np.nan,
        'country': data.get('country', np.nan) if data else np.nan,
    }
   
    # preprocess features, load model, make prediction
    def generate_predictions_stream():
        best_sales = -float('inf')
        optimal_price = None
        all_predictions = []

        for price_point in price_range:
            predicted_sales = -float('inf')

            # add relevant data to new data, then merge with original data (pre-engineering)
            new_data['unitprice'] = price_point
            new_df = pd.DataFrame([new_data])
            df = pd.concat([original_df, new_df], ignore_index=True)

            # runs feature engineering
            df = data_handling.scripts.structure_missing_values(df=df)
            df = data_handling.scripts.handle_feature_engineering(df=df)
            
            # load preprocessor
            target_col = 'sales'
            num_cols, cat_cols = data_handling.scripts.categorize_num_cat_cols(df=df, target_col=target_col)
            preprocessor = data_handling.scripts.handle_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

            # preprocess input features
            y = df[target_col]
            X = df.copy().drop(columns=target_col, axis=1)
            X = preprocessor.fit_transform(X)

            # tensor loader
            input_tensor = torch.tensor(X, dtype=torch.float32)
            
            if model:
                model.eval()
                with torch.no_grad():
                    y_pred = model(input_tensor)
                    y_pred_all_stockcodes = y_pred.cpu().numpy().flatten()
                    y_pred_stockcode = y_pred_all_stockcodes[-1]
                    y_pred_actual_stockcode = np.exp(y_pred_stockcode)

                    main_logger.info(
                        f"\nPrediction for stockcode {stockcode}:\nLogged Sales Predicted: {y_pred_stockcode:.4f}\n Actual Sales Predicted: $ {y_pred_actual_stockcode:,.4f}")
                    
                    predicted_sales = y_pred_actual_stockcode

            elif backup_model:
                y_pred_all_stockcodes = backup_model.predict(X)
                y_pred_stockcode = y_pred_all_stockcodes[-1]
                y_pred_actual_stockcode = np.exp(y_pred_stockcode)
                predicted_sales = y_pred_actual_stockcode

                main_logger.info(
                    f"\nPrediction for stockcode {stockcode}:\nLogged Sales Predicted: {y_pred_stockcode:.4f}\nActual Sales Predicted: $ {y_pred_actual_stockcode:,.4f}")
            

            all_predictions.append({
                "unit_price": float(price_point),
                "predicted_sales": float(predicted_sales)
            })

            if predicted_sales > best_sales:
                best_sales = predicted_sales
                optimal_price = price_point

            current_output = {
                "stockcode": stockcode,
                "unit_price": float(price_point),
                "predicted_sales": float(predicted_sales),
                "optimal_unit_price": float(optimal_price) if optimal_price else float(price_point),
                "max_predicted_sales": float(best_sales) if best_sales else float(predicted_sales),
            }
            
            yield json.dumps(current_output) + '\n'

    return Response(generate_predictions_stream(), mimetype='application/json') 

@app.after_request
def add_header(response):   
    # response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


def handler(event, context):
    import json

    main_logger.info("lambda handler invoked.")

    try:
        load_artifacts()
    except (FileNotFoundError, Exception) as e:
        main_logger.error(f"Failed to load model and artifacts: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to load artifacts'})
        }

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
        load_artifacts()
        load_backup_model()
        serve(app, host='0.0.0.0', port=5002)
        
    else:
        app.run(host='0.0.0.0', port=5002)