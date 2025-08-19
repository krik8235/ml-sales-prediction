import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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



def main_script(target_col: str = 'quantity', is_scale: bool = True, impute_stockcode = False, verbose: bool = False):
    preprocessor = None
    ORIGINAL_DF_PATH = os.environ.get('ORIGINAL_DF_PATH', 'data/original_df.parquet')
    PROCESSED_DF_PATH = os.environ.get('PROCESSED_DF_PATH', 'data/processed_df.parquet')
    X_TEST_PATH = os.environ.get('X_TEST', 'data/x_test_df.parquet')

    # load and save the original data frame in parquet
    df = scripts.load_original_dataframe()
    df.to_parquet(ORIGINAL_DF_PATH, index=False)
    s3_upload(ORIGINAL_DF_PATH)

    # feature engineering + imputation
    df = scripts.structure_missing_values(df=df)
    df = scripts.handle_feature_engineering(df=df)
    scripts.save_df_to_csv(df=df)

    df.to_parquet(PROCESSED_DF_PATH, index=False)
    s3_upload(PROCESSED_DF_PATH)

    # creates imputation data by stockcode (stores parquet files in s3)
    if impute_stockcode: scripts.create_imputation_values_by_stockcode(base_df=df)

    # classify num and cat columns
    num_cols, cat_cols = scripts.categorize_num_cat_cols(df=df, target_col=target_col)
    if verbose: main_logger.info(f'num_cols: {num_cols} \ncat_cols: {cat_cols}')

    if cat_cols:
        for col in cat_cols: df[col] = df[col].astype('string')

    # creates train, val, test datasets
    y = df[target_col]
    X = df.copy().drop(target_col, axis='columns')

    if X.isna().any().any(): main_logger.warning('input X has NaN'); raise Exception()
    if X.isnull().any().any(): main_logger.warning('input X has null'); raise Exception()
    if y.isna().any().any(): main_logger.warning('target y has NaN'); raise Exception()
    if y.isnull().any().any(): main_logger.warning('target y has null'); raise Exception()

    test_size, random_state = 50000, 42
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=test_size, random_state=random_state)

    # store X_test in local and s3 before transforming
    X_test.to_parquet(X_TEST_PATH, index=False)
    s3_upload(X_TEST_PATH)

    if is_scale: X_train, X_val, X_test, preprocessor = scripts.transform_input(X_train, X_val, X_test, num_cols=num_cols, cat_cols=cat_cols)
    else: X_train, X_val, X_test, _ = scripts.transform_input(X_train, X_val, X_test, num_cols=[], cat_cols=cat_cols)

    # y_train, y_val, y_test, target_scaler = transform_target(y_train, y_val, y_test)

    if np.isnan(X_train).any(): main_logger.error("NaNs found in scaled data"); raise Exception()
    if np.isinf(X_train).any(): main_logger.error("Infs found in scaled data"); raise Exception()

    if preprocessor is not None: preprocessor.fit(X)

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor



def main_script_by_stockcode(stockcode: str, target_col: str = 'quantity'):
    if not stockcode: main_logger.error('need stockcode'); raise

    PROCESSED_DF_PATH_STOCKCKODE_CSV = os.environ.get('PROCESSED_DF_PATH', f'data/processed_df_{stockcode}.csv')
    PROCESSED_DF_PATH_STOCKCKODE_PARQUET = os.environ.get('PROCESSED_DF_PATH', f'data/processed_df_{stockcode}.parquet')
    X_TEST_PATH_STOCKCKODE = os.environ.get('X_TEST', f'data/x_test_df_{stockcode}.parquet')
    PREPROCESSOR_PATH = 'preprocessors/column_transformer.pkl'

    # load and save the original data frame in parquet
    df = scripts.load_original_dataframe()
    df = scripts.structure_missing_values(df=df)
    df = scripts.handle_feature_engineering(df=df)

    df_stockcode = df[df['stockcode'] == stockcode]
    if df_stockcode is None: main_logger.error(f'failed to load df by the given stockcode {stockcode}. return the entire processed df'); return df

    df_stockcode.to_csv(PROCESSED_DF_PATH_STOCKCKODE_CSV, index=False)
    df_stockcode.to_parquet(PROCESSED_DF_PATH_STOCKCKODE_PARQUET, index=False)
    s3_upload(PROCESSED_DF_PATH_STOCKCKODE_PARQUET)

    # creates train, val, test datasets
    y = df_stockcode[target_col]
    X = df_stockcode.copy().drop(target_col, axis='columns')

    if X.isna().any().any(): main_logger.warning('input X has NaN'); raise Exception()
    if X.isnull().any().any(): main_logger.warning('input X has null'); raise Exception()
    if y.isna().any().any(): main_logger.warning('target y has NaN'); raise Exception()
    if y.isnull().any().any(): main_logger.warning('target y has null'); raise Exception()

    test_size, random_state = min(int(len(X) * 0.3), 500), 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # store X_test in local and s3 before transforming
    X_test.to_parquet(X_TEST_PATH_STOCKCKODE, index=False)
    s3_upload(X_TEST_PATH_STOCKCKODE)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # X_train, X_val, X_test, preprocessor = scripts.transform_input(X_train, X_val, X_test, num_cols=num_cols, cat_cols=cat_cols)

    # if np.isnan(X_train).any(): main_logger.error("NaNs found in scaled data"); raise Exception()
    # if np.isinf(X_train).any(): main_logger.error("Infs found in scaled data"); raise Exception()

    # if preprocessor is not None: preprocessor.fit(X)

    return X_train, X_test, y_train, y_test, preprocessor
