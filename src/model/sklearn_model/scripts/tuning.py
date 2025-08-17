import numpy as np
import pandas as pd
import lightgbm as lgb  # type: ignore
from typing import Iterable
from itertools import product
from functools import partial
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize # type: ignore

from src.model.sklearn_model.scripts.predict import make_prediction
from src.model.sklearn_model.scripts.saving import save_model_to_local
from src._utils import main_logger

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)


def run_kfold_validation(
        X_train,
        y_train,
        base_model,
        hparams: dict,
        n_splits: int = 5,
        early_stopping_rounds: int = 10,
        max_iters: int = 200
    ) -> float:

    X_train = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    y_train = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train

    mses = 0.0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        if fold == 0: main_logger.info(f'hyperparameters to test: {hparams}')

        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model = base_model(**hparams)

        best_val_mse = float('inf')
        patience_counter = 0
        best_model_state = None
        best_iteration = 0

        if isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
            try:
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    eval_metric='l2',                       # 'l2' for MSE
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                )
                y_pred_val_kf = model.predict(X_val_fold)
                current_val_mse = mean_squared_error(y_val_fold, y_pred_val_kf) # type: ignore

                # check for nan of inf in predictions
                if np.any(np.isnan(y_pred_val_kf)) or np.any(np.isinf(y_pred_val_kf)):  # type: ignore
                    main_logger.warning(f"fold # {fold}: predictions contain NaN or Inf. Returning a high penalty MSE.")
                    mses += 1e10
                else:
                    mses += current_val_mse

            except Exception as e:
                main_logger.error(f"error during LightGBM training in fold #{fold} with hparams {hparams}: {e}")
                mses += 1e10 # add a large penalty for failed training

        else:
            for iteration in range(max_iters):
                try:
                    model.train_one_step(X_train_fold, y_train_fold, iteration)
                except AttributeError:
                    # main_logger.error("Model does not have a 'train_one_step' method for manual early stopping.")
                    model.fit(X_train_fold, y_train_fold)
                    break

                model.fit(X_train_fold, y_train_fold)
                y_pred_val_kf = model.predict(X_val_fold)
                current_val_mse = mean_squared_error(y_val_fold, y_pred_val_kf)

                if current_val_mse < best_val_mse:
                    best_val_mse = current_val_mse
                    patience_counter = 0
                    best_model_state = model.get_params()
                    best_iteration = iteration
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_rounds:
                    main_logger.info(f"Fold {fold}: Early stopping triggered at iteration {iteration} (best at {best_iteration}). Best MSE: {best_val_mse:.4f}")
                    break

            else:
                main_logger.info(f"fold #{fold}: Max iterations ({max_iters}) reached. Best MSE: {best_val_mse:.4f}")

            if best_model_state: model.set_params(**best_model_state)
                # main_logger.info(f"Fold {fold}: Restored model to best state from iteration {best_iteration}.")

        y_pred_val_kf = model.predict(X_val_fold)
        mses += mean_squared_error(y_pred_val_kf, y_val_fold)  # type: ignore

    ave_mse = mses / n_splits
    return ave_mse


# grid search
def _grid_search(X_train, y_train, search_space: dict, base_model) -> tuple[Iterable, dict, float]:
    """Finds optimal hyperparameters via Grid Search and returns trained optimal model, hyperparameters, and best MSE."""

    keys, values = search_space.keys(), search_space.values()
    all_combinations_tuples = list(product(*values))

    all_combinations_list = []
    for combo_tuple in all_combinations_tuples:
        combination_dict = dict(zip(keys, combo_tuple))
        all_combinations_list.append(combination_dict)

    best_mse = float('inf')
    best_hparams = dict()

    for hparams in all_combinations_list:
        current_mse = run_kfold_validation(X_train=X_train, y_train=y_train, base_model=base_model, hparams=hparams)

        if current_mse < best_mse:
            best_mse = current_mse
            best_hparams = hparams

    best_model = base_model(**best_hparams)
    best_model.fit(X_train, y_train)
    main_logger.info(f'best hparams:\n{best_hparams}\nbest MSE {best_mse:.4f}')

    return best_model, best_hparams, best_mse


def run_grid_search(
        X_train, X_val, y_train, y_val,
        search_space: dict,
        base_model,
        model_name: str = 'en'
    ):

    best_model, best_hparams, _ = _grid_search(X_train, y_train, search_space=search_space, base_model=base_model)

    if not best_model and best_hparams:
        best_model = base_model(**best_hparams)

    if best_model:
        main_logger.info('successfully load the model (grid search)')
        match model_name:
            case 'gbm':
                best_model.fit( # type: ignore
                    X_train, y_train,
                    eval_set=[(X_val, y_val)], eval_metric='l2', callbacks=[lgb.early_stopping(10, verbose=False)] # type: ignore
                )
            case _:
                best_model.fit(X_train, y_train) # type: ignore

        _, _, _, rmsle = make_prediction(model=best_model, X=X_val, y=y_val)

    else:
        main_logger.error('failed to complete the grid search. return emply model')
        best_model = ElasticNet() if model_name == 'en' else lgb.LGBMRegressor()

    return best_model, best_hparams, rmsle


# bayesian optimization
def _bayesian_optimization(X_train, y_train, space: list, base_model, n_calls=50) -> tuple[Iterable, dict, float]:
    # @use_named_args(space)
    def objective(params, X_train, y_train, base_model, hparam_names):
        hparams = {item: params[i] for i, item in enumerate(hparam_names)}
        ave_mse = run_kfold_validation(X_train=X_train, y_train=y_train, base_model=base_model, hparams=hparams)
        return ave_mse

    hparam_names = [s.name for s in space]
    objective_partial = partial(objective, X_train=X_train, y_train=y_train, base_model=base_model, hparam_names=hparam_names)
    results = gp_minimize(
        func=objective_partial,
        dimensions=space,
        n_calls=n_calls,
        random_state=42,
        verbose=False,
        n_initial_points=10,
    )
    best_hparams = dict(zip(hparam_names, results.x)) # type: ignore
    best_mse = results.fun # type: ignore
    best_model = base_model(**best_hparams)

    main_logger.info(f'best hparams:\n{best_hparams}\nbest MSE {best_mse:.4f}')

    return best_model, best_hparams, best_mse


def run_bayesian_optimization(
        X_train, X_val, y_train, y_val,
        search_space: list,
        base_model,
        model_name: str = 'en'
    ):

    best_model, best_hparams, _ = _bayesian_optimization(X_train, y_train, space=search_space, base_model=base_model)

    if not best_model and best_hparams:
        best_model = base_model(**best_hparams)

    if best_model:
        main_logger.info('successfully load the model (bayesian optimization)')
        match model_name:
            case 'gbm':
                best_model.fit( # type: ignore
                    X_train, y_train,
                    eval_set=[(X_val, y_val)], eval_metric='l2', callbacks=[lgb.early_stopping(10, verbose=False)] # type: ignore
                )
            case _:
                best_model.fit(X_train, y_train) # type: ignore

        _, _, _, rmsle = make_prediction(model=best_model, X=X_val, y=y_val)

    else:
        main_logger.error('failed to complete bayesian optimization. return emply model')
        best_model = ElasticNet() if model_name == 'en' else lgb.LGBMRegressor()

    return best_model, best_hparams, rmsle
