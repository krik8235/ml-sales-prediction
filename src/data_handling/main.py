
import os
import numpy as np
import pandas as pd

import src.data_handling.scripts as scripts
from src._utils import main_logger, s3_upload


def reconstruct_dataframe(original_df: pd.DataFrame, new_df_to_add: pd.DataFrame = None) -> pd.DataFrame: # type: ignore
    if new_df_to_add is not None:
        df = pd.concat([original_df, new_df_to_add], ignore_index=True)
        df = scripts.structure_missing_values(df=df)
        df = scripts.handle_feature_engineering(df=df)
    
    else:
        df = scripts.structure_missing_values(df=original_df)
        df = scripts.handle_feature_engineering(df=df)       
    return df



def main_script(is_scale: bool = True, verbose: bool = False, should_save: bool = False):
    try:
        df, file_path, file_name = scripts.load_post_feature_engineer_dataframe()
    except:
        df = scripts.load_original_dataframe()
        df = scripts.structure_missing_values(df=df)
        df = scripts.handle_feature_engineering(df=df)
        file_path, file_name = scripts.save_df_to_csv(df=df)
    
    if 'Unnamed: 0' in df.columns: df = df.drop('Unnamed: 0', axis=1)
  
    if should_save:
        S3_DATA_PREFIX = os.environ.get('S3_DATA_PREFIX')
        s3_upload(file_path=file_path, file_name=file_name, prefix=S3_DATA_PREFIX)

    target_col = 'sales'
    num_cols, cat_cols = scripts.categorize_num_cat_cols(df=df, target_col=target_col)
    if verbose: main_logger.info(f'num_cols: {num_cols} \ncat_cols: {cat_cols}')

    # creates datasets
    X_train, X_val, X_test, y_train, y_val, y_test = scripts.make_train_val_datasets(df=df, target_col=target_col, test_size=50000)

    if is_scale is True:
        X_train, X_val, X_test = scripts.transform_input(X_train, X_val, X_test, num_cols=num_cols, cat_cols=cat_cols)
    
    else:
        X_train, X_val, X_test = scripts.transform_input(X_train, X_val, X_test, num_cols=[], cat_cols=cat_cols)
    
    # y_train, y_val, y_test, target_scaler = transform_target(y_train, y_val, y_test)

    if np.isnan(X_train).any(): main_logger.error("NaNs found in scaled data")
    if np.isinf(X_train).any(): main_logger.error("Infs found in scaled data")

    return X_train, X_val, X_test, y_train, y_val, y_test



# with open(MODEL_SAVE_PATH + 'df_base.pkl', 'wb') as f:
#     pickle.dump(df_base, f)
# s3_client.upload_file(MODEL_SAVE_PATH + 'df_base.pkl', S3_BUCKET_NAME, f"{S3_MODEL_PREFIX}df_base.pkl")