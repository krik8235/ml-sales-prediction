import torch.nn as nn
import numpy as np

import src.model.torch_model.scripts as t
from src._utils import main_logger


def main_script(X_train, X_val, y_train, y_val, should_local_save: bool = True, grid: bool = False) -> tuple[nn.Module, dict]:
    """
    Tunes the PyTorch model using GridSearch and Bayesian Optimization.
    Then, stores the search results on optimal models as timestamped files in the local storage.
    Lastly, selects and returns the best performing model out of all search trials.
    The best performing model is retrained on the full input data (X) before saving.
    """

    # bayesian optimization
    main_logger.info('... start bayesian optimization for DFN ...')
    best_dfn, best_optimizer, best_batch_size, best_checkpoint = t.bayesian_optimization(X_train, X_val, y_train, y_val)
    if should_local_save: t.save_model_to_local(checkpoint=best_checkpoint, trig='bayesian')


    # grid search (optional)
    if grid:
        main_logger.info('... start grid search on DFN ...')
        search_space_grid = {
            'learning_rate': [1e-3, 1e-5],
            'batch_size': [32, 64, 128],
            'num_layers': [3, 5],
            'drop'
            'optimizer_name': ['adam', 'sgd']
        }
        best_dfn_grid, optimizer_grid, batch_size_grid, checkpoint_grid = t.grid_search(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            search_space=search_space_grid
        )
        if should_local_save: t.save_model_to_local(checkpoint=checkpoint_grid, trig='grid')

        # validation - find and save best performer
        main_logger.info('... selecting the best performer ...')
        _, _, _, rmsle_test_dfn_grid = t.make_prediction(model=best_dfn_grid, X=X_val, y=y_val)
        _, _, _, rmsle_test_dfn_bayesian = t.make_prediction(model=best_dfn, X=X_val, y=y_val)

        best_dfn = best_dfn_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_dfn
        best_optimizer = optimizer_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_optimizer
        best_batch_size = batch_size_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_batch_size
        best_checkpoint = checkpoint_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_checkpoint


    # retrain the best model w the train + val dataset
    # X, y = np.concatenate([X_train, X_val, X_test], axis=0), np.concatenate([y_train, y_val, y_test], axis=0)
    X_tv, y_tv = np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0)
    best_dfn_full_trained, _ = t.train_model(
        X_train=X_tv, y_train=y_tv, model=best_dfn, optimizer=best_optimizer, batch_size=best_batch_size, num_epochs=1000
    )
    best_checkpoint['model_state_dict'] = best_dfn_full_trained.state_dict()

    # save the retrained best model to local
    if should_local_save: t.save_model_to_local(checkpoint=best_checkpoint, trig='best')
    return best_dfn_full_trained, best_checkpoint
