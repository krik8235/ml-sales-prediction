import os
import sys
import pandas as pd
import torch
import torch.nn as nn

import src.model.torch_model.scripts as scripts
from src._utils import main_logger


def tune_and_train(
        X_train, X_val, y_train, y_val,
        stockcode: str = '',
        should_local_save: bool = True,
        grid: bool = False,
        n_trials: int = 50,
        num_epochs: int = 3000
    ) -> tuple[nn.Module, dict]:

    # bayesian optimization
    main_logger.info('... start bayesian optimization for DFN ...')
    best_dfn, best_optimizer, best_batch_size, best_checkpoint = scripts.bayesian_optimization(
        X_train, X_val, y_train, y_val, n_trials=n_trials, num_epochs=num_epochs
    )
    if should_local_save: scripts.save_model_to_local(checkpoint=best_checkpoint, trig='bayesian')


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
        best_dfn_grid, optimizer_grid, batch_size_grid, checkpoint_grid = scripts.grid_search(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            search_space=search_space_grid
        )
        if should_local_save: scripts.save_model_to_local(checkpoint=checkpoint_grid, trig='grid')

        # validation - find and save best performer
        main_logger.info('... selecting the best performer ...')
        _, _, _, rmsle_test_dfn_grid = scripts.perform_inference(model=best_dfn_grid, X=X_val, y=y_val)
        _, _, _, rmsle_test_dfn_bayesian = scripts.perform_inference(model=best_dfn, X=X_val, y=y_val)

        best_dfn = best_dfn_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_dfn
        best_optimizer = optimizer_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_optimizer
        best_batch_size = batch_size_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_batch_size
        best_checkpoint = checkpoint_grid if rmsle_test_dfn_grid < rmsle_test_dfn_bayesian else best_checkpoint


    if not best_dfn: main_logger.error('... missing model ...'); raise

    # save local if necessary
    if should_local_save: scripts.save_model_to_local(checkpoint=best_checkpoint, trig='best')

    # dvc track - model
    DFN_FILE_PATH = os.path.join('models', 'production', f'dfn_best_{stockcode}.pth' if stockcode else 'dfn_best.pth')
    os.makedirs(os.path.dirname(DFN_FILE_PATH), exist_ok=True)
    torch.save(best_checkpoint, DFN_FILE_PATH)

    return best_dfn, best_checkpoint


def track_metrics_by_stockcode(X_val, y_val, best_model, checkpoint: dict, stockcode: str):
    # dvc track - metrics
    _, mse, exp_mae, rmsle = scripts.perform_inference(model=best_model, X=X_val, y=y_val)
    metrics = dict(mse=mse, mae=exp_mae, rmsle=rmsle)
    scripts.save_metrics(metrics=metrics, filepath=os.path.join('metrics', f'val_{stockcode}.json'))

    # save historical metrics
    model_version = f"dfn_{stockcode}_{os.getpid()}"
    scripts.save_historical_metric_to_s3(
        stockcode=stockcode,
        metrics=metrics,
        model_version=model_version,

        # kwargs from checkpoint
        hparams=checkpoint['hparams'],
        optimizer=checkpoint['optimizer_name'],
        batch_size=checkpoint['batch_size'],
        lr=checkpoint['lr']
    )


if __name__ == '__main__':
    # fetch vals from command args
    X_TRAIN_PATH = sys.argv[1]
    X_VAL_PATH = sys.argv[2]
    Y_TRAIN_PATH = sys.argv[3]
    Y_VAL_PATH = sys.argv[4]
    SHOULD_LOCAL_SAVE = sys.argv[5] == 'True'
    GRID = sys.argv[6] == 'True'
    N_TRIALS = int(sys.argv[7])
    NUM_EPOCHS = int(sys.argv[8])
    STOCKCODE = str(sys.argv[9])

    # extract datasets from dvc cache
    X_train, X_val = pd.read_parquet(X_TRAIN_PATH), pd.read_parquet(X_VAL_PATH)
    y_train, y_val = pd.read_parquet(Y_TRAIN_PATH), pd.read_parquet(Y_VAL_PATH)


    # device type handling
    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device_type == 'cpu':
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    # tuning
    best_model, checkpoint = tune_and_train(
        X_train, X_val, y_train, y_val,
        stockcode=STOCKCODE, should_local_save=SHOULD_LOCAL_SAVE, grid=GRID, n_trials=N_TRIALS, num_epochs=NUM_EPOCHS
    )

    # metrics
    track_metrics_by_stockcode(X_val, y_val, best_model=best_model, checkpoint=checkpoint, stockcode=STOCKCODE)
