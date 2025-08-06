import torch
import torch.nn as nn
import numpy as np

import src.model.torch_model.scripts as t
from src._utils import main_logger


def main_script(X_train, X_val, X_test, y_train, y_val, y_test) -> nn.Module:
    """
    Tunes the PyTorch model using GridSearch and Bayesian Optimization. 
    Then, stores the search results on optimal models as timestamped files in the local storage.
    Lastly, selects and returns the best performing model out of all search trials.
    The best performing model is retrained on the full input data (X) before saving.
    """

    # try:
    # grid search
    main_logger.info('... start grid search on DFN ...')
    search_space_grid = {
        'learning_rate': [1e-3, 1e-5],
        'batch_size': [32, 64, 128],
        'num_layers': [3, 5],
        'optimizer_name': ['adam', 'sgd']
    }
    best_dfn_grid, optimizer_grid, batch_size_grid = t.grid_search(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        search_space=search_space_grid
    )
    t.save_model_to_local(model=best_dfn_grid, trig='grid')


    # bayesian optimization
    main_logger.info('... start bayesian optimization for DFN ...')
    best_dfn_bayesian, optimizer_bayesian, batch_size_bayesian = t.bayesian_optimization(X_train, X_val, y_train, y_val)
    t.save_model_to_local(model=best_dfn_bayesian, trig='bayesian')


    # validation - find and save best performer
    main_logger.info('... selecting the best performer ...')
    _, _, _, rmsle_test_dfn_grid = t.make_prediction(model=best_dfn_grid, X=X_test, y=y_test)
    _, _, _, rmsle_test_dfn_bayesian = t.make_prediction(model=best_dfn_bayesian, X=X_test, y=y_test)

    best_dfn = best_dfn_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_dfn_bayesian
    best_optimizer = optimizer_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else optimizer_bayesian
    best_batch_size = batch_size_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else batch_size_bayesian

    # retrain the best model w the entire dataset
    X, y = np.concatenate([X_train, X_val, X_test], axis=0), np.concatenate([y_train, y_val, y_test], axis=0)
    best_dfn_full_trained, _ = t.train_model(X, y, model=best_dfn, optimizer=best_optimizer, batch_size=best_batch_size, num_epochs=200)

    # save the retrained best model to local
    t.save_model_to_local(model=best_dfn_full_trained, trig='best')
    return best_dfn_full_trained
        
    # except:
    #     main_logger.error('failed to load or run search to find the best dfn. returning empty dfn.')
    #     return t.DFN(input_dim=X_train.shape[1])
