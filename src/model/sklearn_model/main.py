import src.model.sklearn_model as sk
from src._utils import main_logger


def main_script(
        X_train, X_val, y_train, y_val,
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

        # bayesian optimization
        best_model_bayesian, best_hparams_bayesian, rmsle_bayesian = sk.scripts.run_bayesian_optimization(
            X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
            search_space=search_space_bayesian,
            base_model=base_model,
            model_name=model_name,
        )

        best_model = best_model_bayesian if rmsle_bayesian < rmsle_grid else best_model_grid # type: ignore
        best_hparams = best_hparams_bayesian if rmsle_bayesian < rmsle_grid  else best_hparams_grid # type: ignore
        sk.scripts.save_model_to_local(model=best_model, hparams=best_hparams, model_name=model_name, trig='best') # type: ignore

        return best_model, best_hparams
        
    except:
        main_logger.error('failed to load, search, or tune the model.')
        return None, None
        



# def load():
#     best_model, best_hparams = sk.load_model(model_name=model_name, trig='best')
    
#     if best_model is not None:
#         match model_name:
#             case 'gbm':
#                 best_model.fit(
#                     X_train, y_train, 
#                     eval_set=[(X_val, y_val)], eval_metric='l2', callbacks=[lgb.early_stopping(10, verbose=False)] # type: ignore
#                 )
#             case _:
#                 best_model.fit(X_train, y_train)

#         _, mse_best, mae_best, rmsle_best = sk.make_prediction(X=X_val, y=y_val, model=best_model)
#         main_logger.info(
#             f"successfully loaded best performing model {model_name}\nBest hyperparameters: {best_hparams}\nMSE: {mse_best:,.4f}\nMAE: $ {mae_best:,.4f}\nRMSLE: {rmsle_best:,.4f}")
        
#         return best_model