
import numpy as np
import pandas as pd

import src.data_handling.scripts as scripts
from src._utils import main_logger


def reconstruct_dataframe(original_df: pd.DataFrame, new_df_to_add: pd.DataFrame = None) -> pd.DataFrame: # type: ignore
    if new_df_to_add is not None:
        df = pd.concat([original_df, new_df_to_add], ignore_index=True)
        df = scripts.structure_missing_values(df=df)
        df = scripts.handle_feature_engineering(df=df)
    
    else:
        df = scripts.structure_missing_values(df=original_df)
        df = scripts.handle_feature_engineering(df=df)       
    return df



def main_script(is_scale: bool = True, verbose: bool = False):
    try:
        df, _, _ = scripts.load_post_feature_engineer_dataframe()
    except:
        df = scripts.load_original_dataframe()
        df = scripts.structure_missing_values(df=df)
        df = scripts.handle_feature_engineering(df=df)
        _, _ = scripts.save_df_to_csv(df=df)
    
    if 'Unnamed: 0' in df.columns: df = df.drop('Unnamed: 0', axis=1)

    target_col = 'sales'
    num_cols, cat_cols = scripts.categorize_num_cat_cols(df=df, target_col=target_col)
    if verbose: main_logger.info(f'num_cols: {num_cols} \ncat_cols: {cat_cols}')

    # creates datasets
    X_train, X_val, X_test, y_train, y_val, y_test, X, _ = scripts.make_train_val_datasets(df=df, target_col=target_col, test_size=50000)

    if is_scale is True:
        X_train, X_val, X_test, preprocessor = scripts.transform_input(X_train, X_val, X_test, num_cols=num_cols, cat_cols=cat_cols)
    
    else:
        X_train, X_val, X_test, _ = scripts.transform_input(X_train, X_val, X_test, num_cols=[], cat_cols=cat_cols)
    
    # y_train, y_val, y_test, target_scaler = transform_target(y_train, y_val, y_test)

    if np.isnan(X_train).any(): main_logger.error("NaNs found in scaled data")
    if np.isinf(X_train).any(): main_logger.error("Infs found in scaled data")

    if preprocessor: preprocessor.fit(X)
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
