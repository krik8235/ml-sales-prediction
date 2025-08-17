import os
import sys
import joblib
import torch
import warnings
import numpy as np
from dotenv import load_dotenv # type: ignore
from sklearn.model_selection import train_test_split

import src.data_handling as data_handling
import src.model.torch_model as t
from src._utils import s3_upload, main_logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

# paths
PRODUCTION_MODEL_FOLDER_PATH = 'models/production'


if __name__ == '__main__':
    load_dotenv(override=True)

    # explicitly disable multithreaded operation - forcing PyTorch to use a single thread for CPU operations
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.makedirs(PRODUCTION_MODEL_FOLDER_PATH, exist_ok=True)

    stockcode = sys.argv[1] if len(sys.argv) > 1 else None
    if not stockcode: main_logger.error('missing a stockcode'); raise

    # file paths
    DFN_FILE_PATH_OVERALL_BEST = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'dfn_best.pth')
    DFN_FILE_PATH_STOCKCODE = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, f'dfn_best_{stockcode}.pth')
    PREPROCESSOR_PATH = 'preprocessors/column_transformer.pkl'

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    df = data_handling.scripts.load_original_dataframe()
    df = data_handling.scripts.structure_missing_values(df=df)
    df = data_handling.scripts.handle_feature_engineering(df=df)
    df_stockcode = df[df['stockcode'] == stockcode]

    target_col = 'quantity'
    X_stockcode = df_stockcode.copy().drop(columns=target_col)
    y_stockcode = df_stockcode.copy()[target_col]

    test_size, random_state = int(min(len(X_stockcode) * 0.3, 500)), 42  # type: ignore
    X_tv, X_test, y_tv, y_test = train_test_split(X_stockcode, y_stockcode, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=test_size, random_state=random_state)

    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    model, checkpoint = t.main_script(X_train_processed, X_val_processed, y_train, y_val, should_local_save=False)
    torch.save(checkpoint, DFN_FILE_PATH_STOCKCODE)
    s3_upload(file_path=DFN_FILE_PATH_STOCKCODE)

    # batch_size = 16
    # model = t.scripts.load_model(input_dim=X_train.shape[1], file_path=DFN_FILE_PATH_OVERALL_BEST) # type: ignore

    # train_data_loader = t.scripts.create_torch_data_loader(X=X_train, y=y_train, batch_size=batch_size)
    # val_data_loader = t.scripts.create_torch_data_loader(X=X_val, y=y_val, batch_size=batch_size)

    # # retrain the best model
    # retrained_model, _ = t.scripts.train_model(
    #     model=model,
    #     optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    #     criterion=torch.nn.MSELoss(),
    #     num_epochs=1000,
    #     min_delta=0.00001,
    #     patience=10,
    #     train_data_loader=train_data_loader,
    #     val_data_loader=val_data_loader,
    #     device_type='cpu'
    # )

    # torch.save(retrained_model.state_dict(), DFN_FILE_PATH_STOCKCODE)
    # s3_upload(file_path=DFN_FILE_PATH_STOCKCODE)
