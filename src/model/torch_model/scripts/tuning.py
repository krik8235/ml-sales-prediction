import itertools
import pandas as pd
import optuna # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.model.torch_model.scripts.pretrained_base import DFN
from src.model.torch_model.scripts.training import train_model
from src._utils import main_logger



def create_torch_data_loader(X, y, batch_size: int = 32) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts NumPy's ndarray or Pandas' Series data to Tensor Dataset, and then returns the PyTorch DataLoader.
    """
    X_np = X.values if isinstance(X, pd.Series) else X
    X_final_tensor = torch.tensor(X_np, dtype=torch.float32)

    y_np = y.values if isinstance(y, pd.Series) else y
    y_final_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

    dataset_final = TensorDataset(X_final_tensor, y_final_tensor)
    data_loader = DataLoader(dataset_final, batch_size=batch_size, shuffle=True)
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
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer
    learning_rate = hparams.get('learning_rate', backup_hparams.get('learning_rate', 0.00042474594783356774))
    optimizer_name = hparams.get('optimizer_name', backup_hparams.get('optimizer','rmsprop'))
    optimizer = _handle_optimizer(optimizer_name=optimizer_name, model=model, lr=learning_rate)

    return model, optimizer


# def train_model(
#         X_train, X_val, y_train, y_val, 
#         model,
#         batch_size,
#         optimizer,
#         criterion,
#         num_epochs: int = 50,
#         min_delta: float = 1e-5,
#         patience: int = 10
#     ):
#     X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
#         X_train, y_train, test_size=5000, shuffle=True, random_state=42
#     )
#     train_data_loader = create_torch_data_loader(X=X_train_search, y=y_train_search, batch_size=batch_size)
#     val_data_loader = create_torch_data_loader(X=X_val_search, y=y_val_search, batch_size=batch_size)

#     train_data_loader = create_torch_data_loader(X_train, y_train, batch_size=batch_size)
#     val_data_loader = create_torch_data_loader(X=X_val, y=y_val, batch_size=batch_size)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # start training and validation
#     best_val_loss = float('inf')
#     epochs_no_improve = 0
#     for epoch in range(num_epochs):
#         model.train()
#         for batch_X, batch_y in train_data_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#         if (epoch + 1) % 10 == 0: main_logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

#         # validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch_X_val, batch_y_val in val_data_loader:
#                 batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
#                 outputs_val = model(batch_X_val)
#                 val_loss += criterion(outputs_val, batch_y_val).item()

#         val_loss /= len(val_data_loader)
      
#         if val_loss < best_val_loss - min_delta:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 main_logger.info(f'Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.')
#                 break

#     return model, best_val_loss




# def train_model(X_train, y_train, criterion=None, verbose: bool = False, model=None, **kwargs):
#     """
#     Initilizes a DFN (or another torch model) with the optimal hyperparameters and returns the trained model with early stopping.
#     """
    
#     model = model if model else DFN(input_dim=X_train.shape[1])

#     # for training and early-stopping setup
#     batch_size = kwargs.get('batch_size', 16)
#     num_epochs = kwargs.get('num_epochs', 1000)
#     min_delta = kwargs.get('min_delta', 0.0001)
#     patience = kwargs.get('patience', 10)

#     # optimizer
#     learning_rate = kwargs.get('learning_rate', 0.001)
#     optimizer_name = kwargs.get('optimizer', 'adam')
#     optimizer = _handle_optimizer(optimizer_name=optimizer_name, model=model, lr=learning_rate)

#     # train
#     model.train()
#     criterion = criterion if criterion else nn.MSELoss()
    
#     # split training set to train/val sets to detect early stopping
#     X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
#         X_train, y_train, test_size=5000, shuffle=True, random_state=42
#     )
#     train_data_loader = create_torch_data_loader(X=X_train_search, y=y_train_search, batch_size=batch_size)
#     val_data_loader = create_torch_data_loader(X=X_val_search, y=y_val_search, batch_size=batch_size)

#     best_val_loss = float('inf')
#     epochs_no_improve = 0

#     for epoch in range(num_epochs):
#         for batch_X, batch_y in train_data_loader:
#             # zero the gradients
#             optimizer.zero_grad()

#             # forward pass
#             outputs = model(batch_X)           
#             if torch.isnan(outputs).any() or torch.isinf(outputs).any():
#                 main_logger.error(f"Epoch {epoch}: NaN or Inf found in model outputs.")
#                 break
            
#             # backward pass
#             loss = criterion(outputs, batch_y)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#             loss.backward()
#             optimizer.step()

#         if verbose and (epoch + 1) % (num_epochs // 10) == 0 or epoch == num_epochs - 1:
#             main_logger.info(f'Final Training Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch_X_val, batch_y_val in val_data_loader:
#                 val_outputs = model(batch_X_val)
#                 val_loss += criterion(val_outputs, batch_y_val).item()
#         val_loss /= len(val_data_loader)
#         if verbose and (epoch + 1) % (num_epochs // 10) == 0 or epoch == num_epochs - 1:
#             main_logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

#         # add early stopping
#         if val_loss < best_val_loss - min_delta:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 main_logger.info(f'Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.')
#                 break

#     return model



def grid_search(X_train, X_val, y_train, y_val, search_space: dict, verbose: bool = False, criterion=nn.MSELoss()):
    """
    Finds and returns the optimal hyperparameter combination
    """
    best_loss = float('inf')
    best_model = DFN(input_dim=X_train.shape[1])
    best_hparams = {}
    results = []
    keys = search_space.keys()
    combinations = itertools.product(*(search_space[key] for key in keys))

    if verbose:
        main_logger.info("Starting Grid Search...")
        main_logger.info(f"Total combinations to test: {len(list(itertools.product(*(search_space[key] for key in keys))))}\n")

    for i, combo in enumerate(combinations):
        if verbose: main_logger.info(f"--- Testing Combination {i+1} ---")

        current_hparams = dict(zip(keys, combo))
        model, optimizer = construct_model_and_optimizer(hparams=current_hparams, input_dim=X_train.shape[1])
        batch_size = current_hparams.get('batch_size', 16)       
        model, _ = train_model(
            X_train, X_val, y_train, y_val,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=200,
            min_delta=1e-5,
            patience=10
        )
   
        # evaluate
        model.eval()
        data_loader_val = create_torch_data_loader(X=X_val, y=y_val, batch_size=batch_size)
        total_loss, num_batches = 0.0, 0

        with torch.no_grad():        # disable gradient calculation during evaluation
            for batch_X, batch_y in data_loader_val:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        if verbose: main_logger.info(f"Average Loss for this combination: {avg_loss:.4f}\n")

        results.append({'hparams': current_hparams, 'avg_loss': avg_loss })
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_hparams = current_hparams
            best_model = model

    if verbose:
        main_logger.info("\n--- Grid Search Complete ---")
        main_logger.info(f"Best Loss Found: {best_loss:.4f}")
        main_logger.info(f"Best Hyperparameters: {best_hparams}")
        main_logger.info("\nAll Results:")
        for res in results:
            main_logger.info(f"hparams: {res['hparams']}, Avg Loss: {res['avg_loss']:.4f}")

    return best_model



def bayesian_optimization(X_train, X_val, y_train, y_val) -> tuple[nn.Module, dict]:
    """
    Runs Bayesian Optimization to search the best hyperparameters.
    """

    # splits training dataset into train/val datasets.
    input_dim = X_train.shape[1]
    X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
        X_train, y_train, test_size=5000, shuffle=True, random_state=42
    )
    
    # loss function
    criterion = nn.MSELoss()
    
    # define objective for optuna
    def objective(trial):
        # model
        num_layers = trial.suggest_int('num_layers', 1, 5)
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        dropout_rates = []
        hidden_units_per_layer = []
        for i in range(num_layers):
            dropout_rates.append(trial.suggest_float(f'dropout_rate_layer_{i}', 0.0, 0.5))
            hidden_units_per_layer.append(trial.suggest_int(f'n_units_layer_{i}', 16, 128)) # hidden units per layer
        model = DFN(
            input_dim,
            num_layers=num_layers,
            dropout_rates=dropout_rates,
            batch_norm=batch_norm,
            hidden_units_per_layer=hidden_units_per_layer
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # optimizer
        learning_rate = trial.suggest_float('learning_rate', 1e-10, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd', 'adamw', 'adamax', 'adadelta', 'radam'])
        optimizer = _handle_optimizer(optimizer_name=optimizer_name, model=model, lr=learning_rate)

        # loss function
        criterion = nn.MSELoss()

        # train/val data setting
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        train_loader = create_torch_data_loader(X=X_train_search, y=y_train_search, batch_size=batch_size)
        val_loader = create_torch_data_loader(X=X_val_search, y=y_val_search, batch_size=batch_size)

        # epoch
        num_epochs = trial.suggest_int('num_epochs', 10, 500)
        
        # start training & validation
        best_val_loss = float('inf')
        epochs_no_improve = 0
        min_delta = 1e-5
        patience = 10
        for epoch in range(num_epochs):
            model.train()

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model(batch_X_val)
                    val_loss += criterion(outputs_val, batch_y_val).item()
            val_loss /= len(val_loader)

            # early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    main_logger.info(f'Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.')
                    break
            
            trial.report(val_loss, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            
            main_logger.info(f'Epoch {epoch} - Loss: {val_loss:.4f}')
        
        return best_val_loss

    
    # start optimization
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50, timeout=600)
    main_logger.info(f"Number of finished trials: {len(study.trials)}")
    main_logger.info(f"Number of pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")

    # construct best model and best optimizer
    best_trial = study.best_trial
    best_hparams = best_trial.params
    main_logger.info(
        f"\nBest trial found (Trial #{best_trial.number}):\nValidation MSE: {best_trial.value:.4f}\nBest Hyperparameters: {best_hparams}")

    best_lr = best_hparams['learning_rate']
    best_batch_size = best_hparams['batch_size']
    best_model = DFN(
        input_dim=input_dim,
        num_layers=best_hparams['num_layers'],
        hidden_units_per_layer=[v for k, v in best_hparams.items() if 'n_units_layer_' in k],
        batch_norm=best_hparams['batch_norm'],
        dropout_rates=[v for k, v in best_hparams.items() if 'dropout_rate_layer_' in k],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    best_optimizer_name = best_hparams['optimizer']
    best_optimizer = _handle_optimizer(optimizer_name=best_optimizer_name, model=best_model, lr=best_lr)
    

    # retrain the best model with full training dataset (use optimal batch size)   
    num_epochs = 50
    full_train_dataset_loader = create_torch_data_loader(X_train, y_train, batch_size=best_batch_size)
    for epoch in range(num_epochs):
        best_model.train()
        for batch_X, batch_y in full_train_dataset_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            best_optimizer.zero_grad()
            outputs = best_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            best_optimizer.step()
        if (epoch + 1) % 5 == 0: main_logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


    # validate the best model with validation dataset
    best_model.eval()
    val_loss = 0.0
    val_loader = create_torch_data_loader(X=X_val, y=y_val, batch_size=best_batch_size)

    # suspend gradient computation. no need to run optimizer
    with torch.no_grad():
        for batch_X_val, batch_y_val in val_loader:
            batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
            outputs_val = best_model(batch_X_val)
            val_loss += criterion(outputs_val, batch_y_val).item()
    val_loss /= len(val_loader)
    main_logger.info(f"\nFinal Model Evaluation on Validation Data (MSE): {val_loss:.4f}")

    return best_model, best_hparams