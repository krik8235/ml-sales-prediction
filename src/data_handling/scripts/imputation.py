import os
import numpy as np
import pandas as pd

from src._utils import main_logger, s3_upload

ORIGINAL_DF_PATH = os.environ.get('ORIGINAL_DF_PATH', 'data/original_df.parquet')


def structure_missing_values(df: pd.DataFrame, target_cols_to_impute: list = [], verbose: bool = False) -> pd.DataFrame:
    pd.set_option('future.no_silent_downcasting', True)

    if df is None or not isinstance(df, pd.DataFrame):
        main_logger.error("missing dataframe. need a pandas dataframe")
        return None

    target_cols_to_impute = target_cols_to_impute if target_cols_to_impute else df.columns.tolist()

    df_processed = df.copy()
    unstructured_nan_vals = ['', 'nan', 'N/A', None, 'na', 'None', 'none']
    structured_nan = { item: np.nan for item in unstructured_nan_vals }

    for col in target_cols_to_impute:
        df_processed[col].replace(structured_nan)

    if verbose: main_logger.info(f'structured missing data into NaN. {df_processed.isna().sum()}')
    return df_processed


def fetch_imputation_cache_key_and_file_path(stockcode):
    cache_key = f"imp_data:{stockcode}"
    temp_file_path = f'/tmp/stockcode_{stockcode}.parquet'
    return cache_key, temp_file_path


def create_imputation_values_by_stockcode(stockcode=None):
    os.makedirs('tmp', exist_ok=True)
    original_df = pd.read_parquet(ORIGINAL_DF_PATH)

    if stockcode is None:
        imputation_df = original_df.groupby('stockcode').agg(
            unitprice_median=('unitprice', 'median'),
            unitprice_max=('unitprice', 'max'),
            unitprice_min=('unitprice', 'min'),
            quantity_mean=('quantity', 'mean'),
            country=('country', lambda x: x.mode()[0])
        )
        for st_code, data in imputation_df.iterrows():
            _, temp_file_path = fetch_imputation_cache_key_and_file_path(stockcode=st_code)
            df = pd.DataFrame([data])
            df.to_parquet(temp_file_path, index=False)
            s3_upload(temp_file_path)
            os.remove(temp_file_path)

    else:
        df = original_df[original_df['stockcode'] == stockcode].agg(
            unitprice_median=('unitprice', 'median'),
            unitprice_max=('unitprice', 'max'),
            unitprice_min=('unitprice', 'min'),
            quantity_mean=('quantity', 'mean'),
            country=('country', lambda x: x.mode()[0])
        )
        _, temp_file_path = fetch_imputation_cache_key_and_file_path(stockcode=stockcode)
        df.to_parquet(temp_file_path, index=False)
        s3_upload(temp_file_path)
        os.remove(temp_file_path)


if __name__ == "__main__":
    create_imputation_values_by_stockcode()
