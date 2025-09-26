import os
import sys
import torch
from dotenv import load_dotenv

import src.data_handling as data_handling
import src.model.torch_model as t
from src._utils import s3_upload, main_logger


def main_script_stockcode(stockcode: str):
    load_dotenv(override=True)

    # file paths
    PRODUCTION_MODEL_FOLDER_PATH = os.path.join('models', 'production')
    os.makedirs(PRODUCTION_MODEL_FOLDER_PATH, exist_ok=True)

    DFN_FILE_PATH_STOCKCODE = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, f'dfn_best_{stockcode}.pth')

    X_train, X_val, _, y_train, y_val, _, _ = data_handling.main_script_by_stockcode(stockcode=stockcode)

    _, checkpoint = t.main_script(X_train, X_val, y_train, y_val, should_local_save=False, n_trials=100)
    torch.save(checkpoint, DFN_FILE_PATH_STOCKCODE)
    s3_upload(file_path=DFN_FILE_PATH_STOCKCODE)



if __name__ == '__main__':
    # fetch stockcode
    stockcode = sys.argv[1] if len(sys.argv) > 1 else None
    if not stockcode: main_logger.error('missing a stockcode'); raise

    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device_type == 'cpu':
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    main_script_stockcode(stockcode=stockcode)
