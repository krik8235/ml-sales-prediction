import pandas as pd

import src.data_handling.scripts as scripts
from src.data_handling.etl_pipeline import etl_pipeline
from src.data_handling.preprocess import preprocess
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



def main_script(target_col: str = 'quantity', should_scale: bool = True, impute_stockcode: bool = False, verbose: bool = False):
    etl_pipeline(impute_stockcode=impute_stockcode)

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess(
        target_col=target_col, should_scale=should_scale, verbose=verbose
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


def main_script_by_stockcode(stockcode: str, target_col: str = 'quantity'):
    if not stockcode: main_logger.error('need stockcode'); raise

    etl_pipeline(stockcode=stockcode)

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess(
        target_col=target_col, should_scale=True, verbose=False, stockcode=stockcode
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
