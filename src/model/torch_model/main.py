import torch.nn as nn

import src.model.torch_model.scripts as t
from src._utils import main_logger


def main_script(X_train, X_val, X_test, y_train, y_val, y_test) -> nn.Module:
    """loads or trains best torch model and returns the traind model."""
    
    try:
        best_dfn = t.load_model(input_dim=X_train.shape[1])
        if best_dfn:
            main_logger.info('successfully load best performing DFN.')
            return best_dfn
        
        else:
            # grid search 
            best_dfn_grid = t.load_model(input_dim=X_train.shape[1], trig='grid')

            if not best_dfn_grid:
                main_logger.info('... start grid search on DFN ...')
                search_space_grid = {
                    'learning_rate': [1e-3, 1e-5],
                    'batch_size': [32, 64, 128],
                    'num_layers': [3, 5],
                    'optimizer_name': ['adam', 'sgd']
                }
                best_dfn_grid = t.grid_search(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    search_space=search_space_grid
                )
                t.save_model(model=best_dfn_grid, trig='grid')

            # bayesian opt
            best_dfn_bayesian = t.load_model(input_dim=X_train.shape[1],  trig='bayesian')

            if not best_dfn_bayesian:
                main_logger.info('... start bayesian optimization for DFN ...')
                best_dfn_bayesian, _ = t.bayesian_optimization(X_train, X_val, y_train, y_val)
                t.save_model(model=best_dfn_bayesian, trig='bayesian')


            # validation - find and save best performer
            main_logger.info('Start validation to select best performer.')
            _, _, _, rmsle_test_dfn_grid = t.make_prediction(model=best_dfn_grid, X=X_test, y=y_test)
            _, _, _, rmsle_test_dfn_bayesian = t.make_prediction(model=best_dfn_bayesian, X=X_test, y=y_test)

            best_dfn = best_dfn_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_dfn_bayesian
            t.save_model(model=best_dfn, trig='best')
    
            return best_dfn
        
    except:
        main_logger.error('failed to load or run search to find the best dfn. returning empty dfn.')
        return t.DFN(input_dim=X_train.shape[1])
