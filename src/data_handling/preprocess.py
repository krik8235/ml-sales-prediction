import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import src.data_handling.scripts as scripts
from src._utils import main_logger



def preprocess(stockcode: str = '', target_col: str = 'quantity', should_scale: bool = True, verbose: bool = False):
    # integrate w/ dvc.yaml to extract the processed df
    df = pd.read_parquet(f'data/processed_df_{stockcode}.parquet' if stockcode else f'data/processed_df.parquet')

    # categorize num and cat columns
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

    random_state = 42
    test_size = int(min(len(X) * 0.2, 500)) if stockcode else 50000
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=test_size, random_state=random_state)

    # store train, val, test data
    X_train.to_parquet(f'data/x_train_df_{stockcode}.parquet' if stockcode else 'data/x_train_df.parquet', index=False)
    X_val.to_parquet(f'data/x_val_df_{stockcode}.parquet' if stockcode else 'data/x_val_df.parquet', index=False)
    X_test.to_parquet(f'data/x_test_df_{stockcode}.parquet' if stockcode else 'data/x_test_df.parquet', index=False)
    y_train.to_frame(name=target_col).to_parquet(
        f'data/y_train_df_{stockcode}.parquet' if stockcode else 'data/y_train_df.parquet',
        index=False
    )
    y_val.to_frame(name=target_col).to_parquet(
        f'data/y_val_df_{stockcode}.parquet' if stockcode else 'data/y_val_df.parquet',
        index=False
    )
    y_test.to_frame(name=target_col).to_parquet(
        f'data/y_test_df_{stockcode}.parquet' if stockcode else 'data/y_test_df.parquet',
        index=False
    )

    # s3_upload(X_TEST_PATH) ## intentionally comment out (no longer need to store data in s3 separately)


    preprocessor = None
    PREPROCESSOR_PATH = 'preprocessors/column_transformer.pkl'

    if should_scale:
        X_train, X_val, X_test, preprocessor = scripts.transform_input(X_train, X_val, X_test, num_cols=num_cols, cat_cols=cat_cols)
    else:
        X_train, X_val, X_test, _ = scripts.transform_input(X_train, X_val, X_test, num_cols=[], cat_cols=cat_cols)

    if np.isnan(X_train).any(): main_logger.error('NaNs found in scaled data'); raise
    if np.isinf(X_train).any(): main_logger.error('Infs found in scaled data'); raise

    if not stockcode and preprocessor is not None:
        preprocessor.fit(X)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

    return  X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


if __name__ == '__main__':
    stockcode_list = ['85123A', ]

    parser = argparse.ArgumentParser(description='run data preprocessing')
    parser.add_argument('--stockcode', type=str, default='', help='specific stockcode')
    parser.add_argument('--target_col', type=str, default='quantity', help='the target column name')
    parser.add_argument('--should_scale', type=bool, default=True, help='flag to scale numerical features')
    parser.add_argument('--verbose', type=bool, default=False, help='flag for verbose logging')
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess(
        target_col=args.target_col,
        should_scale=args.should_scale,
        verbose=args.verbose
    )
