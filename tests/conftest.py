import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock
from dotenv import load_dotenv # type: ignore

# load env and adjust path
load_dotenv(override=True)
os.environ['CORS_ORIGINS'] = 'http://localhost:3000, http://127.0.0.1:3000'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# import app after setting mock paths
import app
from src._utils import main_logger


@pytest.fixture(autouse=True) # applies the fixture to all tests automatically
def mock_external_dependencies(monkeypatch):

    main_logger.info("... mocking external dependencies for pytest run ...")

    # mock the s3_load_to_temp_file function to prevent s3 access
    def mock_s3_load_to_temp_file(file_path):
        main_logger.info(f"Mocked s3_load_to_temp_file called for: {file_path}")
        return "/tmp/mock_file" # Return a dummy path

    # mock the s3_load function
    def mock_s3_load(file_path):
        main_logger.info(f"Mocked s3_load called for: {file_path}")
        return MagicMock() # Return a mock object

    # mock the boto3 client to prevent log errors about missing credentials
    def mock_boto3_client(service_name, region_name=None):
        main_logger.info(f"Mocked boto3.client called for service: {service_name}")
        mock_client = MagicMock()
        if service_name == 's3':
            mock_client.head_object.return_value = {} # Pretend the object exists
        elif service_name == 'sts':
            mock_client.get_caller_identity.return_value = {'Arn': 'mock_arn'}
        return mock_client


    # mock the joblib.load to prevent file system access
    def mock_joblib_load(file_path):
        main_logger.info(f"Mocked joblib.load called for: {file_path}")
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        return mock_preprocessor

    # apply all the mocks using monkeypatch
    monkeypatch.setattr(app, 's3_load_to_temp_file', mock_s3_load_to_temp_file)
    monkeypatch.setattr(app, 's3_load', mock_s3_load)
    monkeypatch.setattr(app.joblib, 'load', mock_joblib_load)
    monkeypatch.setattr('boto3.client', mock_boto3_client)


@pytest.fixture
def mock_df():
    df = pd.DataFrame({
        'stockcode': ['85123A', '85123A', 'B', 'B'],
        'quantity': [100, 200, 50, 75],
        'unitprice': [2.5, 3.0, 5.0, 6.0],
        'unitprice_min': [2.0, 2.0, 4.0, 4.0],
        'unitprice_median': [2.8, 2.8, 5.5, 5.5],
        'unitprice_max': [4.0, 4.0, 7.0, 7.0]
    })
    return df


@pytest.fixture
def mock_preprocessor():
    """Fixture for a mock preprocessor, now just a simple object."""
    mock = MagicMock()
    mock.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    return mock


@pytest.fixture
def mock_torch_model():
    """Fixture for a mock PyTorch model."""
    mock = MagicMock()
    mock.return_value = torch.tensor([[np.log(150)]], dtype=torch.float32)
    mock.eval.return_value = None
    return mock


@pytest.fixture
def mock_backup_model():
    """Fixture for a mock backup model (e.g., GBM)."""
    mock = MagicMock()
    mock.predict.return_value = np.array([np.log(120)])
    mock.feature_name_ = ['a', 'b', 'c']
    mock.coef_ = [1, 2, 3]
    return mock


@pytest.fixture
def flask_client():
    """Fixture for the Flask test client."""
    app.app.config['TESTING'] = True
    with app.app.test_client() as client:
        yield client


# mock redis client - returns mock data
class MockRedisClient:
    def get(self, key: str = 'predict-price-85123A'):
        if key == 'predict-price-85123A':
            cached_data = [{"stockcode": "85123A", "predicted_sales": 999.9}]
            return json.dumps(cached_data)
        return None

    def ping(self):
        return True

    def set(self, *args, **kwargs):
        return True

    def flushall(self, *args, **kwargs):
        return True

    def keys(self, *args, **kwargs):
        return ['mock_key_1', 'mock_key_2']

    def delete(self, *args, **kwargs):
        return 2


@pytest.fixture
def mock_redis_client():
    main_logger.info("... fixture `mock_redis_client` is executed ...")
    return MockRedisClient()


# patch the get_redis_client function in app.py to return our mock
@pytest.fixture(autouse=True)
def patch_redis_client_for_tests(monkeypatch, mock_redis_client):
    def mock_get_redis_client():
        return mock_redis_client
    monkeypatch.setattr(app, 'get_redis_client', mock_get_redis_client)
