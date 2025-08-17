import torch
import torch.nn as nn
import optuna # type: ignore
from sklearn.model_selection import train_test_split

from src._utils import main_logger


def train_model(
        model,
        optimizer,
        criterion = nn.MSELoss(),
        num_epochs: int = 50,
        min_delta: float = 1e-3,
        patience: int = 10,
        trial=None,
        train_data_loader=None, val_data_loader=None,
        X_train=None, y_train=None, batch_size=32, # backup args when data loaders are not given
        device_type=None,
    ) -> tuple[nn.Module, float]:

    from src.model.torch_model.scripts.tuning import create_torch_data_loader

    # device
    device_type = device_type if device_type else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_type)

    # gradient scaler for stability (only for cuba)
    scaler = torch.GradScaler(device=device_type) if device_type == 'cuba' else None

    if train_data_loader is None or val_data_loader is None:
    # if not np.all(train_data_loader) or not np.all(val_data_loader):
        # set up train/validation data loader
        try:
            X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(X_train, y_train, test_size=10000, random_state=42)
            train_data_loader = create_torch_data_loader(X=X_train_search, y=y_train_search, batch_size=batch_size)
            val_data_loader = create_torch_data_loader(X=X_val_search, y=y_val_search, batch_size=batch_size)
        except:
            main_logger.error('need a data loader or training dataset. return empty model')
            return model, float('inf')


    # start training - with validation and early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        main_logger.info(f'... start epoch {epoch + 1} ...')
        model.train()
        for batch_X, batch_y in train_data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            try:
                # pytorch's AMP system automatically handles the casting of tensors to Float16 for operations that benefit from it (like the large matrix multiplications in nn.Linear layers) and keeps them in Float32 for operations that might suffer from reduced precision (like nn.BatchNorm1d often does, or the loss calculation).
                with torch.autocast(device_type=device_type):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                        main_logger.error('pytorch model returns nan or inf. break the training loop.')
                        break

                # create scaled gradients of the loss
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # cliping grad
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)  # unscales the gradients. call optimizer.step if gradient is not inf or nan
                    scaler.update()  # updates the scale

                else:
                    loss.backward()
                    # cliping grad
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            except:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        if (epoch + 1) % 10 == 0: main_logger.info(f"epoch [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}")

        # validate on a validation dataset (subset of the entire training dataset)
        model.eval()
        val_loss = 0.0

        # switch the grad mode (suspend grad computation)
        with torch.inference_mode():
            for batch_X_val, batch_y_val in val_data_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                outputs_val = model(batch_X_val)
                val_loss += criterion(outputs_val, batch_y_val).item()

        val_loss /= len(val_data_loader)

        # early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                main_logger.info(f'early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.')
                break

        # for optuna (bayesian opt for hparams tuning)
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    return model, best_val_loss
