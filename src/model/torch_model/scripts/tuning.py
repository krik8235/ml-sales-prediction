import itertools
import pandas as pd
import numpy as np
import optuna # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.model.torch_model.scripts.pretrained_base import DFN
from src.model.torch_model.scripts.training import train_model
from src._utils import main_logger


device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)


def create_torch_data_loader(X, y, batch_size: int = 32) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts NumPy's ndarray or Pandas' Series data to Tensor Dataset and returns the PyTorch DataLoader.
    """

    X_np = X.values if isinstance(X, pd.Series) else X.to_numpy() if isinstance(X, pd.DataFrame) else X

    # use lower-precision of float16 for latency
    X_final_tensor = torch.tensor(X_np, dtype=torch.float32)

    y_np = y.values if isinstance(y, pd.Series) else y.to_numpy() if isinstance(y, pd.DataFrame)  else y
    y_final_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

    dataset_final = TensorDataset(X_final_tensor, y_final_tensor)
    data_loader = DataLoader(dataset_final, batch_size=batch_size)
    return data_loader # type: ignore [return-value]


def _handle_optimizer(optimizer_name: str, model: nn.Module, lr: float, **kwargs):
    """
    An utility function to define and return the optimizer in torch.optim format.
    """

    optimizer = None
    match optimizer_name.lower():
        case 'adam': optimizer = optim.Adam(model.parameters(), lr=lr, **kwargs)
        case 'adamw': optimizer = optim.AdamW(model.parameters(), lr=lr, **kwargs)
        case 'adamax': optimizer = optim.Adamax(model.parameters(), lr=lr, **kwargs)
        case 'adadelta': optimizer = optim.Adadelta(model.parameters(), lr=lr, **kwargs)
        case 'adafactor': optimizer = optim.Adafactor(model.parameters(), lr=lr, **kwargs)
        case 'rmsprop': optimizer = optim.RMSprop(model.parameters(), lr=lr, **kwargs)
        case 'radam': optimizer = optim.RAdam(model.parameters(), lr=lr, **kwargs)
        case 'rprop': optimizer = optim.Rprop(model.parameters(), lr=lr, **kwargs)
        case 'sgd': optimizer = optim.SGD(model.parameters(), lr=lr, **kwargs)
        case _: optimizer = optim.Adam(model.parameters(), lr=lr, **kwargs)
    return optimizer



def construct_model_and_optimizer(input_dim, hparams = dict()):
    backup_hparams = {'num_layers': 5, 'batch_norm': False, 'dropout_rate_layer_0': 0.3153784999786484, 'n_units_layer_0': 16, 'dropout_rate_layer_1': 0.2874085374275699, 'n_units_layer_1': 83, 'dropout_rate_layer_2': 0.08096495562953843, 'n_units_layer_2': 122, 'dropout_rate_layer_3': 0.4593903897613426, 'n_units_layer_3': 38, 'dropout_rate_layer_4': 0.4279282102729623, 'n_units_layer_4': 49, 'learning_rate': 0.00042474594783356774, 'optimizer': 'rmsprop', 'batch_size': 64, 'num_epochs': 462}

    num_layers = hparams.get('num_layers', backup_hparams.get('num_layer', 5))
    batch_norm = hparams.get('batch_norm', backup_hparams.get('batch_norm', 5))

    hidden_units_per_layer = [v for k, v in hparams.items() if 'n_units_layer_' in k]
    if not hidden_units_per_layer:
        hidden_units_per_layer=[v for k, v in backup_hparams.items() if 'n_units_layer_' in k]
        if not hidden_units_per_layer: hidden_units_per_layer = [64] * num_layers

    dropout_rates=[v for k, v in hparams.items() if 'dropout_rate_layer_' in k]
    if not dropout_rates:
        dropout_rates=[v for k, v in backup_hparams.items() if 'dropout_rate_layer_' in k]
        if not dropout_rates: dropout_rates = [0.1 for _ in range(0, num_layers)]

    model = DFN(
        input_dim=input_dim,
        num_layers=num_layers,
        hidden_units_per_layer=hidden_units_per_layer,
        batch_norm=batch_norm,
        dropout_rates=dropout_rates,
    ).to(device)

    # optimizer
    learning_rate = hparams.get('learning_rate', backup_hparams.get('learning_rate', 0.00042474594783356774))
    optimizer_name = hparams.get('optimizer_name', backup_hparams.get('optimizer','rmsprop'))
    optimizer = _handle_optimizer(optimizer_name=optimizer_name, model=model, lr=learning_rate)

    return model, optimizer



def grid_search(X_train, X_val, y_train, y_val, search_space: dict, verbose: bool = True, criterion=nn.MSELoss()):
    """
    Finds and returns the optimal hyperparameter combination
    """
    best_loss = float('inf')
    best_model = DFN(input_dim=X_train.shape[1])
    best_hparams = {}
    results = []
    keys = search_space.keys()
    combinations = itertools.product(*(search_space[key] for key in keys))

    main_logger.info(f"grid search: total {len(list(itertools.product(*(search_space[key] for key in keys))))} combinations to test")

    for i, combo in enumerate(combinations):
        if verbose: main_logger.info(f"testing combination {i+1}: {combo}")

        current_hparams = dict(zip(keys, combo))
        model, optimizer = construct_model_and_optimizer(hparams=current_hparams, input_dim=X_train.shape[1])
        batch_size = current_hparams.get('batch_size', 16)

        # set up train/validation data loader
        test_size = 10000 if len(X_train) > 15000 else int(len(X_train) * 0.2)
        X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
        train_data_loader = create_torch_data_loader(X=X_train_search, y=y_train_search, batch_size=batch_size)
        val_data_loader = create_torch_data_loader(X=X_val_search, y=y_val_search, batch_size=batch_size)

        # train the model
        model, val_loss = train_model(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=50,
            min_delta=1e-5,
            patience=10
        )

        # evaluation on validation dataset
        model.eval()
        data_loader_val = create_torch_data_loader(X=X_val, y=y_val, batch_size=batch_size)
        total_loss, num_batches = 0.0, 0

        # switch the grad mode (suspend grad computation)
        with torch.inference_mode():
            for batch_X, batch_y in data_loader_val:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        if verbose: main_logger.info(f"average loss: {avg_loss:.4f}\n")

        results.append({'hparams': current_hparams, 'avg_loss': val_loss })
        if val_loss < best_loss:
            best_loss = val_loss
            best_hparams = current_hparams
            best_model = model

    main_logger.info(f" ... grid search complete ...\nbest validation loss: {best_loss:.4f}\nbest hyperparameters: {best_hparams}")
    main_logger.info("\nall results:")
    for res in results: main_logger.info(f"hparams: {res['hparams']}, avg loss: {res['avg_loss']:.4f}")

    checkpoint = {
        'state_dict': best_model.state_dict(),
        'hparams': best_hparams,
        'input_dim': X_train.shape[1],
        'optimizer': optimizer,
        'batch_size': batch_size
    }
    return best_model, optimizer, batch_size, checkpoint



def bayesian_optimization(X_train, X_val, y_train, y_val):
    """
    Runs Bayesian Optimization to search the best hyperparameters.
    """
    # loss function
    criterion = nn.MSELoss()


    # define objective function for optuna
    def objective(trial):
        # model
        num_layers = trial.suggest_int('num_layers', 1, 20)
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        dropout_rates = []
        hidden_units_per_layer = []
        for i in range(num_layers):
            dropout_rates.append(trial.suggest_float(f'dropout_rate_layer_{i}', 0.0, 0.6))
            hidden_units_per_layer.append(trial.suggest_int(f'n_units_layer_{i}', 8, 256)) # hidden units per layer

        model = DFN(
            input_dim=X_train.shape[1],
            num_layers=num_layers,
            dropout_rates=dropout_rates,
            batch_norm=batch_norm,
            hidden_units_per_layer=hidden_units_per_layer
        ).to(device)

        # optimizer
        learning_rate = trial.suggest_float('learning_rate', 1e-10, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd', 'adamw', 'adamax', 'adadelta', 'radam'])
        optimizer = _handle_optimizer(optimizer_name=optimizer_name, model=model, lr=learning_rate)

        # data loaders
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        test_size = 10000 if len(X_train) > 15000 else int(len(X_train) * 0.2)
        X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
        train_data_loader = create_torch_data_loader(X=X_train_search, y=y_train_search, batch_size=batch_size)
        val_data_loader = create_torch_data_loader(X=X_val_search, y=y_val_search, batch_size=batch_size)

        # training
        # num_epochs = trial.suggest_int('num_epochs', 500, 1000)
        num_epochs = 3000
        _, best_val_loss = train_model(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            model=model,
            optimizer=optimizer,
            criterion = criterion,
            num_epochs=num_epochs,
            trial=trial,
        )
        return best_val_loss


    # start optimization
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50, timeout=600)
    main_logger.info(f"Number of finished trials: {len(study.trials)}")
    main_logger.info(f"Number of pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")

    # best found
    best_trial = study.best_trial
    best_hparams = best_trial.params

    main_logger.info(f" ... bayesian optimization completed ... \nbest found in trial #{best_trial.number}\nbest validation loss: {best_trial.value:.4f}\nbest hyperparameteres: {best_hparams}")

    # construct best model and best optimizer
    best_lr = best_hparams['learning_rate']
    best_batch_size = best_hparams['batch_size']
    input_dim = X_train.shape[1]
    best_model = DFN(
        input_dim=input_dim,
        num_layers=best_hparams['num_layers'],
        hidden_units_per_layer=[v for k, v in best_hparams.items() if 'n_units_layer_' in k],
        batch_norm=best_hparams['batch_norm'],
        dropout_rates=[v for k, v in best_hparams.items() if 'dropout_rate_layer_' in k],
    ).to(device)

    # construct best optimizer
    best_optimizer_name = best_hparams['optimizer']
    best_optimizer = _handle_optimizer(optimizer_name=best_optimizer_name, model=best_model, lr=best_lr)

    # data loaders
    train_data_loader = create_torch_data_loader(X=X_train, y=y_train, batch_size=best_batch_size)
    val_data_loader = create_torch_data_loader(X=X_val, y=y_val, batch_size=best_batch_size)

    # retrain the best model with full training dataset (using optimal batch size)
    best_model, _ = train_model(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        model=best_model,
        optimizer=best_optimizer,
        criterion = criterion,
        num_epochs=1000
    )
    checkpoint = {
        'state_dict': best_model.state_dict(),
        'hparams': best_hparams,
        'input_dim': X_train.shape[1],
        'optimizer': best_optimizer,
        'batch_size': best_batch_size
    }
    return best_model, best_optimizer, best_batch_size, checkpoint
