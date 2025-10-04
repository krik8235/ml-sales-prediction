import os
import sys
import torch
from dotenv import load_dotenv

import src.data_handling as data_handling
import src.model.torch_model as torch_model
from src._utils import main_logger



def main_script_stockcode(stockcode: str, n_trials: int = 100):
    load_dotenv(override=True)

    # process datasets
    X_train, X_val, _, y_train, y_val, _, _ = data_handling.main_script_by_stockcode(stockcode=stockcode)

    # tune model
    best_model, checkpoint = torch_model.tune_and_train(X_train, X_val, y_train, y_val, should_local_save=False, n_trials=n_trials)

    # track metrics
    torch_model.track_metrics_by_stockcode(X_val, y_val, stockcode=stockcode, best_model=best_model, checkpoint=checkpoint)


if __name__ == '__main__':
    # fetch stockcode
    stockcode = sys.argv[1] if len(sys.argv) > 1 else None

    if not stockcode: main_logger.error('missing a stockcode'); raise

    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device_type == 'cpu':
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    main_script_stockcode(stockcode=stockcode)
