import os
import json
import io
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# import scripts to test
import app

# add cors origin
os.environ['CORS_ORIGINS'] = 'http://localhost:3000, http://127.0.0.1:3000'


def test_hello_world(flask_client):
    """test the root endpoint in a local environment."""

    with patch.dict(os.environ, {'ENV': 'local'}):
        res = flask_client.get('/')
        assert res.status_code == 200
        assert b'Hello, world' in res.data


@patch('app.t.scripts.load_model')
@patch('torch.load')
@patch('app._redis_client', new_callable=MagicMock)
@patch('app.joblib.load')
@patch('app.s3_load_to_temp_file')
@patch('app.s3_load')
def test_predict_endpoint_primary_model(
    mock_s3_load,
    mock_s3_load_to_temp_file,
    mock_joblib_load,
    mock_redis_client,
    mock_torch_load,
    mock_load_model,
    flask_client,
):
    """test a prediction from the primary model without cache hit."""

    # mock return values for file loading
    mock_preprocessor = MagicMock()
    mock_joblib_load.return_value = mock_preprocessor
    mock_s3_load.return_value = io.BytesIO(b'dummy_data')
    mock_s3_load_to_temp_file.return_value = 'dummy_path'

    # config redis cache for cache miss
    mock_redis_client.get.return_value = None

    # config the model and torch mock
    mock_torch_model = MagicMock()
    mock_load_model.return_value = mock_torch_model
    mock_torch_load.return_value = {'state_dict': 'dummy'}

    # mock model's prediction array
    num_rows = 1200
    num_bins = 100
    expected_length = num_rows * num_bins
    mock_prediction_array = np.random.uniform(1.0, 10.0, size=expected_length)

    # mock the return chain for the model's forward pass
    mock_torch_model.return_value.cpu.return_value.numpy.return_value.flatten.return_value = mock_prediction_array

    # create a mock dataframe
    mock_df_expanded = pd.DataFrame({
        'stockcode': ['85123A'] * num_rows,
        'quantity': np.random.randint(50, 200, size=num_rows),
        'unitprice': np.random.uniform(1.0, 10.0, size=num_rows),
        'unitprice_min': np.random.uniform(1.0, 3.0, size=num_rows),
        'unitprice_median': np.random.uniform(4.0, 6.0, size=num_rows),
        'unitprice_max': np.random.uniform(8.0, 12.0, size=num_rows),
    })

    # set global variables used by the app endpoint
    app.X_test = mock_df_expanded.drop(columns='quantity')
    app.preprocessor = mock_preprocessor

    with patch.object(pd, 'read_parquet', return_value=mock_df_expanded):
        response = flask_client.get('/v1/predict-price/85123A')

    # assertion
    assert response.status_code == 200

    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) == num_bins
    assert data[0]['stockcode'] == '85123A'
    assert 'predicted_sales' in data[0]



@patch('app.s3_load')
@patch('app.s3_load_to_temp_file')
@patch('joblib.load')
@patch('app.get_redis_client')
@patch('app.load_model')
@patch('app.t.scripts')
def test_endpoint_backup_model(
    mock_t_scripts,
    mock_load_model,
    mock_get_redis_client,
    mock_joblib_load,
    mock_s3_load_to_temp_file,
    mock_s3_load,
    flask_client,
):
    """test a prediction from the backup model without cache hit."""

    # mock file io and preprocessor loading
    mock_s3_load.return_value = io.BytesIO(b'dummy_data')
    mock_s3_load_to_temp_file.return_value = 'dummy_path'
    mock_preprocessor = MagicMock()
    mock_joblib_load.return_value = mock_preprocessor

    # mock redis client and cache data
    mock_redis_client = MagicMock()
    mock_redis_client.get.return_value = None
    mock_get_redis_client.return_value = mock_redis_client

    # create mock dataframes for x_test and prediction
    num_rows = 1200
    num_bins = 100
    mock_df_expanded = pd.DataFrame({
        'stockcode': ['85123A'] * num_rows,
        'quantity': np.random.randint(50, 200, size=num_rows),
        'unitprice': np.random.uniform(1.0, 10.0, size=num_rows),
        'unitprice_min': np.random.uniform(1.0, 3.0, size=num_rows),
        'unitprice_median': np.random.uniform(4.0, 6.0, size=num_rows),
        'unitprice_max': np.random.uniform(8.0, 12.0, size=num_rows),
    })

    # mock the backup model and its prediction
    mock_backup_model = MagicMock()
    mock_backup_model.predict.return_value = np.random.uniform(1.0, 5.0, size=num_rows * num_bins)
    mock_backup_model.feature_name_ = ['feat1', 'feat2', 'feat3', 'feat4']
    mock_backup_model.coef_ = np.random.rand(4)

    # set global variables and trigger the fallback logic
    app.X_test = pd.DataFrame(np.random.rand(num_rows, 4), columns=[f'feat{i+1}' for i in range(4)])
    app.X_test['unitprice'] = np.random.uniform(1.0, 10.0, size=num_rows)
    app.preprocessor = mock_preprocessor
    app.model = None
    app.backup_model = mock_backup_model

    # set the side effect on the load_model attribute of the mocked scripts module
    mock_t_scripts.load_model.side_effect = RuntimeError("Primary model loading failed.")

    # mock the behavior of preprocessor transform to return a numpy array
    def mock_transform_side_effect(df):
        return np.random.rand(df.shape[0], len(mock_backup_model.feature_name_))
    mock_preprocessor.transform.side_effect = mock_transform_side_effect # failed

    with patch.object(pd, 'read_parquet', return_value=mock_df_expanded):
        response = flask_client.get('/v1/predict-price/85123A')

    # assertions
    assert response.status_code == 200

    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) == num_bins
    assert data[0]['stockcode'] == '85123A'
    assert 'predicted_sales' in data[0]

    # verify that the loading model is attempted once
    mock_load_model.assert_called_once()



@patch('app._redis_client', new_callable=MagicMock)
@patch('app.s3_load')
@patch('app.s3_load_to_temp_file')
@patch('joblib.load')
def test_predict_endpoint_with_cache_hit(
    mock_joblib_load,
    mock_s3_load_to_temp_file,
    mock_s3_load,
    mock_redis_client,
    flask_client,
):
    # config return vals
    mock_joblib_load.return_value = 'mock_preprocessor'
    mock_s3_load.return_value = io.BytesIO(b'dummy_data')
    mock_s3_load_to_temp_file.return_value = 'dummy_path'

    # config the 'get' method of the global redis client (mock)
    cached_data = json.dumps([{"stockcode": "85123A", "predicted_sales": 999.9}])
    mock_redis_client.get.return_value = cached_data

    res = flask_client.get('/v1/predict-price/85123A')
    assert res.status_code == 200

    data = json.loads(res.data)
    assert isinstance(data, list)
    assert data[0]['predicted_sales'] == 999.9
