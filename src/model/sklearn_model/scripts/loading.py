import os
import pickle
import lightgbm as lgb # type: ignore
from sklearn.linear_model import ElasticNet

from src._utils import main_logger, retrieve_file_path, MODEL_SAVE_PATH


def _pickle_loader(folder_path=None, file_path=None):
    """Loads .pkl file"""

    if file_path is None and folder_path:
        file_path, _ =  retrieve_file_path(folder_path=folder_path)

    if file_path:
        with open(file_path, 'rb') as f:
            loaded_results = pickle.load(f)
            best_model = loaded_results['best_model']
            best_hparams = loaded_results['best_hparams']
            return best_model, best_hparams
    else:
        main_logger.error('Couldnt find a file path. Return None.')
        return None, None


def load_model(model_name: str = 'gbm', trig: str = 'best'):
    """Loads the latest sklearn model from the local folder or the S3 bucket."""

    model, hparams = None, None
    folder_path = os.path.join(MODEL_SAVE_PATH, f'{model_name}_{trig}')

    try:
        model, hparams = _pickle_loader(folder_path=folder_path)

        if not model and hparams:
            match model_name:
                case 'gbm':
                    model = lgb.LGBMRegressor(**hparams)
                case 'en':
                    model = ElasticNet(**hparams)
                case _:
                    pass

        if not model:
            raise Exception('Model not found')
        else:
            main_logger.info("scikit-learn model loaded for retraining.")
            return model, hparams

    except Exception as e:
        main_logger.error(f"failed to load scikit-learn model for retraining: {e}. raise error.")
        raise e
