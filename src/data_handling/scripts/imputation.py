import os
import numpy as np
import pandas as pd

from src._utils import main_logger, s3_upload


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
    cache_key = f"imp_data:{{{stockcode}}}"
    temp_file_path = f'/tmp/stockcode_{stockcode}.parquet'
    return cache_key, temp_file_path



def _create_imputation_df(original_df, stockcode):
    df_stockcode = original_df.copy()[original_df['stockcode'] == stockcode]

    # make sure unitprice is numeric
    df_stockcode['unitprice'] = pd.to_numeric(df_stockcode['unitprice'], errors='coerce')

    customer_latest_by_stockcode = df_stockcode.copy().groupby('customerid', as_index=False).agg(
        customer_recency_days_latest=('customer_recency_days', lambda x: x.iloc[-1]),
        customer_total_spend_ltm_latest=('customer_total_spend_ltm', lambda x: x.iloc[-1]),
        customer_freq_ltm_latest=('customer_freq_ltm', lambda x: x.iloc[-1]),
    )
    imputation_df = pd.DataFrame({
        'unitprice_median': [df_stockcode['unitprice'].median()],
        'unitprice_max': [df_stockcode['unitprice'].max()],
        'unitprice_min': [df_stockcode['unitprice'].min()],
        # 'quantity_mean': [df_stockcode['quantity'].mean()],
        'country': [df_stockcode['country'].mode()[0]],
        'product_avg_sales_last_month': [df_stockcode['product_avg_sales_last_month'].iloc[-1]]
    })
    merged_df = pd.merge(imputation_df, customer_latest_by_stockcode, how='cross')
    return merged_df


def create_imputation_values_by_stockcode(base_df, stockcode=None):
    if stockcode is None:
        stockcodes = base_df['stockcode'].unique()

        for st in stockcodes:
            imputation_df = _create_imputation_df(original_df=base_df, stockcode=st)
            _, temp_file_path = fetch_imputation_cache_key_and_file_path(stockcode=st)
            imputation_df.to_parquet(temp_file_path, index=False)
            s3_upload(temp_file_path)
            os.remove(temp_file_path)

    else:
        imputation_df = _create_imputation_df(original_df=base_df, stockcode=stockcode)
        _, temp_file_path = fetch_imputation_cache_key_and_file_path(stockcode=stockcode)
        imputation_df.to_parquet(temp_file_path, index=False)
        s3_upload(temp_file_path)
        os.remove(temp_file_path)



if __name__ == "__main__":
    os.makedirs('tmp', exist_ok=True)
    ORIGINAL_DF_PATH = os.environ.get('ORIGINAL_DF_PATH', 'data/original_df.parquet')
    PROCESSED_DF_PATH = os.environ.get('PROCESSED_DF_PATH', 'data/processed_df.parquet')

    original_df = pd.read_parquet(ORIGINAL_DF_PATH)
    processed_df = pd.read_parquet(PROCESSED_DF_PATH)

    create_imputation_values_by_stockcode(base_df=processed_df)
