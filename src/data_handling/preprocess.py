import os
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import src.data_handling.scripts as scripts
from src._utils import main_logger



def preprocess(stockcode: str = '', target_col: str = 'quantity', should_scale: bool = True, verbose: bool = False):
    # load processed df from dvc cache
    PROCESSED_DF_PATH = os.path.join('data', 'processed_df.parquet')
    df = pd.read_parquet(PROCESSED_DF_PATH)

    # categorize num and cat columns
    num_cols, cat_cols = scripts.categorize_num_cat_cols(df=df, target_col=target_col)
    if verbose: main_logger.info(f'num_cols: {num_cols} \ncat_cols: {cat_cols}')

    # structure cat cols
    if cat_cols:
        for col in cat_cols: df[col] = df[col].astype('string')

    # initiate preprocessor
    PREPROCESSOR_PATH = os.path.join('preprocessors', 'column_transformer.pkl')
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except:
        preprocessor = scripts.create_preprocessor(num_cols=num_cols if should_scale else [], cat_cols=cat_cols)

    if not stockcode:
        # creates train, val, test datasets
        y = df[target_col]
        X = df.copy().drop(target_col, axis='columns')

        # raise error if nan or inf in datasets
        if X.isna().any().any(): main_logger.error('input X has NaN'); raise
        if X.isnull().any().any(): main_logger.error('input X has null'); raise
        if y.isna().any().any(): main_logger.error('target y has NaN'); raise
        if y.isnull().any().any(): main_logger.error('target y has null'); raise

        # split
        test_size, random_state = 50000, 42
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=test_size, random_state=random_state, shuffle=False)

        # store train, val, test data for model training and inference
        X_train.to_parquet('data/x_train_df.parquet', index=False)
        X_val.to_parquet('data/x_val_df.parquet', index=False)
        X_test.to_parquet('data/x_test_df.parquet', index=False)
        y_train.to_frame(name=target_col).to_parquet('data/y_train_df.parquet', index=False)
        y_val.to_frame(name=target_col).to_parquet('data/y_val_df.parquet', index=False)
        y_test.to_frame(name=target_col).to_parquet('data/y_test_df.parquet', index=False)

        # preprocess
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)

        if np.isnan(X_train).any(): main_logger.error('NaNs found in scaled data'); raise  # type: ignore
        if np.isinf(X_train).any(): main_logger.error('Infs found in scaled data'); raise  # type: ignore

        # dvc track
        pd.DataFrame(X_train).to_parquet(f'data/x_train_processed.parquet', index=False) # type: ignore
        pd.DataFrame(X_val).to_parquet(f'data/x_val_processed.parquet', index=False) # type: ignore
        pd.DataFrame(X_test).to_parquet(f'data/x_test_processed.parquet', index=False) # type: ignore


    else:
        df_stockcode = pd.read_parquet(f'data/processed_df_{stockcode}.parquet')

        y = df_stockcode[target_col]
        X = df_stockcode.copy().drop(target_col, axis='columns')

        test_size, random_state =  int(min(len(X) * 0.2, 500)), 42
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=test_size, random_state=random_state, shuffle=False)

        # store train, val, test data for model training and inference
        X_train.to_parquet(f'data/x_train_df_{stockcode}.parquet', index=False)
        X_val.to_parquet(f'data/x_val_df_{stockcode}.parquet', index=False)
        X_test.to_parquet(f'data/x_test_df_{stockcode}.parquet', index=False)
        y_train.to_frame(name=target_col).to_parquet(f'data/y_train_df_{stockcode}.parquet', index=False)
        y_val.to_frame(name=target_col).to_parquet(f'data/y_val_df_{stockcode}.parquet', index=False)
        y_test.to_frame(name=target_col).to_parquet(f'data/y_test_df_{stockcode}.parquet', index=False)

        # preprocess
        X_train = preprocessor.transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)

        # dvc track
        pd.DataFrame(X_train).to_parquet(f'data/x_train_processed_{stockcode}.parquet', index=False) # type: ignore
        pd.DataFrame(X_val).to_parquet(f'data/x_val_processed_{stockcode}.parquet', index=False) # type: ignore
        pd.DataFrame(X_test).to_parquet(f'data/x_test_processed_{stockcode}.parquet', index=False) # type: ignore


    # upload trained preprocessor
    if should_scale:
        X_full = df.copy().drop(target_col, axis='columns')
        preprocessor.fit(X_full)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

        # save feature names (dvc track) for shap
        with open('preprocessors/feature_names.json', 'w') as f:
            feature_names = preprocessor.get_feature_names_out()
            json.dump(feature_names.tolist(), f)

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
        verbose=args.verbose,
        stockcode=args.stockcode,
    )
