import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder # type: ignore

from src._utils import main_logger



def create_preprocessor(num_cols, cat_cols):
    os.makedirs('preprocessors', exist_ok=True)

    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('encoder', BinaryEncoder(cols=cat_cols))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor


def transform_input(
        X_train, X_val, X_test,
        num_cols: list = list(),
        cat_cols: list = list(),
        verbose: bool = True,
    ) -> tuple:

    preprocessor = create_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    if verbose:
        main_logger.info(f'transformed input datasets: X_train: {X_train_processed.shape}, X_val: {X_val_processed.shape}, X_test: {X_test_processed.shape}')

    # raise error if nan or inf in the preprocessed data
    try:
        if np.isnan(X_train_processed).any(): # type: ignore
            nan_rows, nan_cols = np.where(np.isnan(X_train_processed))  # type: ignore
            raise Exception(f"NaN found in X_train_processed\nsamples: rows={nan_rows[:5]}, cols={nan_cols[:5]}")

        if np.isinf(X_train_processed).any():  # type: ignore
            inf_rows, inf_cols = np.where(np.isinf(X_train_processed)) # type: ignore
            raise Exception(f"Inf found in X_train_processed\nsamples: rows={inf_rows[:5]}, cols={inf_cols[:5]}")
    except:
        pass

    return  X_train_processed, X_val_processed, X_test_processed, preprocessor


def transform_target(y_train, y_val, y_test, scaler=None) -> tuple:
    scaler = scaler if scaler else StandardScaler()

    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_val_reshaped = y_val.values.reshape(-1, 1)
    y_test_reshaped = y_test.values.reshape(-1, 1)

    y_train_processed = scaler.fit_transform(y_train_reshaped)
    y_val_processed = scaler.transform(y_val_reshaped)
    y_test_processed = scaler.transform(y_test_reshaped)

    return y_train_processed, y_val_processed, y_test_processed, scaler


def categorize_num_cat_cols(df, target_col: str = 'quantity') -> tuple[list, list]:
    cat_cols, num_cols = list(), list()

    if df is not None:
        for col in df.columns:
            if col != target_col:
                n_unique_vals = df[col].nunique()
                dtype = df[col].dtype
                match dtype:
                    case 'int64' | 'int32' | 'float64' | 'float32':
                        if col in ['is_registered', 'is_return', 'year', 'year_month']: cat_cols.append(col)
                        else: num_cols.append(col)
                    case 'object' | 'category':
                        cat_cols.append(col)

    return num_cols, cat_cols
