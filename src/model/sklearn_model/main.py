import numpy as np

import src.model.sklearn_model as sk
from src._utils import main_logger


def main_script(
        X_train, X_val, X_test, y_train, y_val, y_test,
        base_model,
        model_name: str = 'en',
        search_space_grid: dict = {},
        search_space_bayesian: list = [],
    ):
    """Loads the trained best performing sklearn model."""

    try:
        # grid search
        best_model_grid, best_hparams_grid, rmsle_grid = sk.scripts.run_grid_search(
            X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
            search_space=search_space_grid,
            base_model=base_model,
            model_name=model_name,
        )
        sk.scripts.save_model_to_local(model=best_model_grid, hparams=best_hparams_grid, model_name=model_name, trig='grid')

        # bayesian optimization
        best_model_bayesian, best_hparams_bayesian, rmsle_bayesian = sk.scripts.run_bayesian_optimization(
            X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
            search_space=search_space_bayesian,
            base_model=base_model,
            model_name=model_name,
        )
        sk.scripts.save_model_to_local(model=best_model_bayesian, hparams=best_hparams_bayesian, model_name=model_name, trig='bayesian') # type: ignore

        # select the best of all
        best_model = best_model_bayesian if rmsle_bayesian < rmsle_grid else best_model_grid # type: ignore
        best_hparams = best_hparams_bayesian if rmsle_bayesian < rmsle_grid  else best_hparams_grid # type: ignore

        # retrain the best model with the entire dataset (X, y)
        match model_name:
            case 'gbm':
                import lightgbm as lgb # type: ignore

                X_tv, y_tv = np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0)
                best_model.fit( # type: ignore
                    X_tv, y_tv,
                    eval_set=[(X_test, y_test)],  # type: ignore
                    eval_metric='l2', # type: ignore
                    callbacks=[lgb.early_stopping(10, verbose=False)] # type: ignore
                )
            case _:
                X, y = np.concatenate([X_train, X_val, X_test], axis=0), np.concatenate([y_train, y_val, y_test], axis=0)
                best_model.fit(X, y) # type: ignore

        # save the retrained model to local
        sk.scripts.save_model_to_local(model=best_model, hparams=best_hparams, model_name=model_name, trig='best') # type: ignore

        return best_model, best_hparams

    except:
        main_logger.error('failed to load, search, or tune the model.')
        return None, None
