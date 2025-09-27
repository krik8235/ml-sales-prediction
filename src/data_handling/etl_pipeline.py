import os
import argparse

import src.data_handling.scripts as scripts
from src._utils import main_logger


def etl_pipeline(stockcode: str = '', impute_stockcode = False): # type: ignore
    # extract the entire data
    df = scripts.extract_original_dataframe()

    # load perquet file
    ORIGINAL_DF_PATH = os.path.join('data', 'original_df.parquet')
    df.to_parquet(ORIGINAL_DF_PATH, index=False) # dvc versioned

    # transform
    df = scripts.structure_missing_values(df=df)
    df = scripts.handle_feature_engineering(df=df)

    if impute_stockcode: scripts.create_imputation_values_by_stockcode(base_df=df)

    # for stockcode specific df
    if stockcode:
        df_stockcode  = df[df['stockcode'] == stockcode]
        if df_stockcode is None: main_logger.error(f'failed to load df by the given stockcode {stockcode}. return the entire processed df')
        df = df if df_stockcode is None else df_stockcode

        # load
        PROCESSED_DF_PATH = os.path.join('data', f'processed_df_{stockcode}.parquet')
        df.to_parquet(PROCESSED_DF_PATH, index=False) # dvc versioned

    # load
    else:
        PROCESSED_DF_PATH = os.path.join('data', 'processed_df.parquet')
        df.to_parquet(PROCESSED_DF_PATH, index=False) # dvc versioned
        # scripts.load_df_to_csv(df=df)
        # s3_upload(PROCESSED_DF_PATH) ## explictly comment out manual s3 upload
    return df


if __name__ == '__main__':
    stockcode_list = ['85123A', ]

    parser = argparse.ArgumentParser(description="run etl pipeline")
    parser.add_argument('--stockcode', type=str, default='', help="specific stockcode to process. empty runs full pipeline.")
    parser.add_argument('--impute', action='store_true', help="flag to create imputation values")
    args = parser.parse_args()

    etl_pipeline(stockcode=args.stockcode, impute_stockcode=args.impute)
