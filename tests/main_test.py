import os
import shutil
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import src.main as main_script


# define paths for test files
PRODUCTION_MODEL_FOLDER_PATH = main_script.PRODUCTION_MODEL_FOLDER_PATH
DFN_FILE_PATH = main_script.DFN_FILE_PATH
GBM_FILE_PATH = main_script.GBM_FILE_PATH
SVR_FILE_PATH = main_script.SVR_FILE_PATH
EN_FILE_PATH = main_script.EN_FILE_PATH
PREPROCESSOR_PATH = main_script.PREPROCESSOR_PATH



@pytest.fixture(autouse=True) # ensure the fixture is always active
def mock_joblib_dump():
    """mocks joblib.dump to prevent picklingerror"""
    with patch('src.main.joblib.dump', new_callable=MagicMock) as mock_dump:
        yield mock_dump


@pytest.fixture(autouse=True)
def mock_pickle_dump():
    """mocks pickle.dump to prevent picklingerror"""
    with patch('src.main.pickle.dump', new_callable=MagicMock) as mock_dump:
        yield mock_dump


@pytest.fixture(autouse=True)
def setup_and_teardown_mock_dirs():
    """creates a mock directory structure for testing and cleans it up."""

    if not os.path.exists(PRODUCTION_MODEL_FOLDER_PATH):
        os.makedirs(PRODUCTION_MODEL_FOLDER_PATH, exist_ok=True)

    if not os.path.exists(os.path.dirname(PREPROCESSOR_PATH)):
        os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    yield

    # cleanup after the test
    if os.path.exists(PRODUCTION_MODEL_FOLDER_PATH):
        shutil.rmtree(PRODUCTION_MODEL_FOLDER_PATH)

    if os.path.exists(os.path.dirname(PREPROCESSOR_PATH)):
        shutil.rmtree(os.path.dirname(PREPROCESSOR_PATH))



@pytest.fixture
def mock_s3_upload():
    """mocks the s3_upload function to prevent actual uploads"""
    with patch('src.main.s3_upload', new_callable=MagicMock) as mock_s3:
        yield mock_s3

@pytest.fixture
def mock_data_handling():
    """mocks the data_handling.main_script to return dummy data and a mock preprocessor"""

    with patch('src.main.data_handling.main_script') as mock_dh:
        mock_preprocessor = MagicMock()

        # create small numpy arrays to simulate data (200 samples and 10 features for x_train)
        X_train_mock = np.random.rand(200, 10)
        y_train_mock = np.random.rand(200)
        X_val_mock = np.random.rand(50, 10)
        y_val_mock = np.random.rand(50)
        X_test_mock = np.random.rand(50, 10)
        y_test_mock = np.random.rand(50)

        mock_dh.return_value = (
            X_train_mock,
            X_val_mock,
            X_test_mock,
            y_train_mock,
            y_val_mock,
            y_test_mock,
            mock_preprocessor
        )
        yield mock_dh


@pytest.fixture
def mock_model_scripts():
    """mocks the model training scripts to return dummy models and checkpoints."""

    with patch('src.main.t.main_script') as mock_torch_script, \
         patch('src.main.sk.main_script') as mock_sklearn_script:

        # mocks pytorch scripts
        mock_dfn = MagicMock()
        mock_checkpoint = { 'model_state_dict': 'mock_state' }
        mock_torch_script.return_value = (mock_dfn, mock_checkpoint)

        # mocks sklearn scripts
        mock_sklearn_script.return_value = (
            MagicMock(), # best_model
            {'param1': 1} # best_hparams
        )

        yield mock_torch_script, mock_sklearn_script


def test_data_loading_and_preprocessor_saving(mock_data_handling, mock_s3_upload, mock_joblib_dump):
    """tests that data loading is called and the preprocessor is saved and uploaded."""

    main_script.main_script()

    # verify that data_handling.main_script was called
    mock_data_handling.assert_called_once()

    # verify preprocessor is dumped in mock file
    mock_joblib_dump.assert_called_once_with(mock_data_handling.return_value[-1], PREPROCESSOR_PATH)

    # verify preprocessor is uploaded to mock s3
    mock_s3_upload.assert_any_call(file_path=PREPROCESSOR_PATH)



def test_model_optimization_and_saving(mock_data_handling, mock_model_scripts, mock_s3_upload):
    """tests that each model's optimization script is called and the results are saved and uploaded."""

    mock_torch_script, mock_sklearn_script = mock_model_scripts
    main_script.main_script()

    # verify each model's main_script was called
    assert mock_torch_script.called
    assert mock_sklearn_script.call_count == len(main_script.sklearn_models)

    # verify that each model file exists and s3_upload was called for it
    ## dfn
    assert os.path.exists(DFN_FILE_PATH)
    mock_s3_upload.assert_any_call(file_path=DFN_FILE_PATH)

    ## svr model
    assert os.path.exists(SVR_FILE_PATH)
    mock_s3_upload.assert_any_call(file_path=SVR_FILE_PATH)

    ## elastic net
    assert os.path.exists(EN_FILE_PATH)
    mock_s3_upload.assert_any_call(file_path=EN_FILE_PATH)

    ## light gbm
    assert os.path.exists(GBM_FILE_PATH)
    mock_s3_upload.assert_any_call(file_path=GBM_FILE_PATH)
