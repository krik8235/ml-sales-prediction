import os
import argparse

import src.data_handling.scripts as scripts
from src._utils import main_logger, s3_upload


def etl_pipeline(stockcode: str = '', impute_stockcode = False):
    df = scripts.extract_original_dataframe()

    if not stockcode:
        ORIGINAL_DF_PATH = os.environ.get('ORIGINAL_DF_PATH', 'data/original_df.parquet')
        df.to_parquet(ORIGINAL_DF_PATH, index=False)
        s3_upload(ORIGINAL_DF_PATH)

        df = scripts.structure_missing_values(df=df)
        df = scripts.handle_feature_engineering(df=df)
        scripts.load_df_to_csv(df=df)

        if impute_stockcode: scripts.create_imputation_values_by_stockcode(base_df=df)

    else:
        df = scripts.structure_missing_values(df=df)
        df = scripts.handle_feature_engineering(df=df)
        df_stockcode  = df[df['stockcode'] == stockcode]
        if df_stockcode is None: main_logger.error(f'failed to load df by the given stockcode {stockcode}. return the entire processed df')

        df = df_stockcode if df_stockcode is not None else df

    PROCESSED_DF_PATH = f'data/processed_df_{stockcode}.parquet' if stockcode else 'data/processed_df.parquet'
    df.to_parquet(PROCESSED_DF_PATH, index=False)
    s3_upload(PROCESSED_DF_PATH)
    return df


if __name__ == '__main__':
    stockcode_list = ['85123A', ]

    parser = argparse.ArgumentParser(description="Run ETL pipeline.")
    parser.add_argument('--stockcode', type=str, default='', help="specific stockcode to process. empty runs full pipeline.")
    parser.add_argument('--impute', action='store_true', help="flag to create imputation values")
    args = parser.parse_args()

    etl_pipeline(stockcode=args.stockcode, impute_stockcode=args.impute)
