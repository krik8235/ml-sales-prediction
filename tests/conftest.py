from dotenv import load_dotenv # type: ignore
load_dotenv(override=True)

# adjust the path to import the main scripts
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import json
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock

import app
from src._utils import main_logger

@pytest.fixture
def mock_df():
    """Fixture to create a mock DataFrame for testing."""
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
    """Fixture to create a mock preprocessor with a transform method."""
    mock = MagicMock()
    mock.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    return mock


@pytest.fixture
def mock_torch_model():
    """Fixture for a mock PyTorch model."""
    mock = MagicMock()
    # ensure the mock model returns a tensor for inference
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
    # ensure the flask app is created once and used for all tests
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


@pytest.fixture
def mock_redis_client():
    """A pytest fixture that returns a mock Redis client."""

    main_logger.info("... fixture `mock_redis_client` is executed ...")
    return MockRedisClient()
