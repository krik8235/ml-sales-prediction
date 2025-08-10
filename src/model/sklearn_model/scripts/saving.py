import pickle

from src._utils import main_logger, create_file_path


def save_model_to_local(model, hparams: dict, model_name: str = 'gbm', trig: str = 'best', **kwargs):
    """Saves trained sklearn models to local and S3 bucket."""

    file_path, _ = create_file_path(model_name=model_name, trig=trig)

    with open(file_path, 'wb') as f:
        pickle.dump({'best_model': model, 'best_hparams': hparams, **kwargs}, f)
        main_logger.info(f"scikit-learn model saved to {file_path}")

    return file_path
