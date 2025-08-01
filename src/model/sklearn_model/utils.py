import src.model.sklearn_model.scripts as sk
from src._utils import main_logger


def load_or_run_grid_search(
        X_train, X_val, y_train, y_val,
        search_space: dict,
        base_model,
        model_name: str = 'en'
    ):

    best_model, best_hparams, rmsle = None, None, None

    try:
        best_model, best_hparams = sk.load_model(model_name=model_name, trig='grid')
    except:
        best_model, best_hparams, _ = sk.grid_search(X_train, y_train, search_space=search_space, base_model=base_model)

    try:
        best_model = best_model if best_model else base_model(**best_hparams) if best_hparams else None
        if best_model:
            match model_name:
                case 'gbm':
                    best_model.fit( # type: ignore
                        X_train, y_train, 
                        eval_set=[(X_val, y_val)], eval_metric='l2', callbacks=[lgb.early_stopping(10, verbose=False)] # type: ignore
                    )
                case _:
                    best_model.fit(X_train, y_train) # type: ignore

            _, _, _, rmsle = sk.make_prediction(model=best_model, X=X_val, y=y_val)

            sk.save_model(model=best_model, hparams=best_hparams, model_name=model_name, trig='grid') # type: ignore
 
    except:
        main_logger.error('Couldnt complete the grid search. return emply model')

    
    return best_model, best_hparams, rmsle



def load_or_run_bayesian_optimization(
        X_train, X_val, y_train, y_val, 
        search_space: list, 
        base_model, 
        model_name: str = 'en'
    ):

    best_model, best_hparams, rmsle = None, None, None

    try:
        best_model, best_hparams = sk.load_model(model_name=model_name, trig='bayesian') 
    except:
        best_model, best_hparams, _ = sk.bayesian_optimization(X_train, y_train, space=search_space, base_model=base_model)

    try:
        best_model = best_model if best_model else base_model(**best_hparams) if best_hparams else None
        if best_model:
            match model_name:
                case 'gbm':
                    best_model.fit( # type: ignore
                        X_train, y_train, 
                        eval_set=[(X_val, y_val)], eval_metric='l2', callbacks=[lgb.early_stopping(10, verbose=False)] # type: ignore
                    )
                case _:
                    best_model.fit(X_train, y_train) # type: ignore

            _, _, _, rmsle = sk.make_prediction(model=best_model, X=X_val, y=y_val)

            sk.save_model(model=best_model, hparams=best_hparams, model_name=model_name, trig='bayesian') # type: ignore
       
    except:
        main_logger.error('Couldnt complete bayesian optimization. Return emply model')
    
 
    return best_model, best_hparams, rmsle
