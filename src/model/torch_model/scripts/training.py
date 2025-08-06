import torch
import torch.nn as nn
import optuna # type: ignore
from sklearn.model_selection import train_test_split

from src._utils import main_logger


def train_model(
        X_train, y_train,
        model,
        batch_size,
        optimizer,
        criterion = nn.MSELoss(),
        num_epochs: int = 50,
        min_delta: float = 1e-5,
        patience: int = 10,
        trial=None
    ) -> tuple[nn.Module, float]:

    from src.model.torch_model.scripts.tuning import create_torch_data_loader

    # device 
    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)
    
    # set up train/validation data loader
    train_data_loader, val_data_loader = None, None
    X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
        X_train, y_train, test_size=10000, shuffle=True, random_state=42
    )
    train_data_loader = create_torch_data_loader(X=X_train_search, y=y_train_search, batch_size=batch_size)
    val_data_loader = create_torch_data_loader(X=X_val_search, y=y_val_search, batch_size=batch_size)

    
    # start training - with validation and early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            try:
                # pytorch's AMP system automatically handles the casting of tensors to Float16 for operations that benefit from it (like the large matrix multiplications in nn.Linear layers) and keeps them in Float32 for operations that might suffer from reduced precision (like nn.BatchNorm1d often does, or the loss calculation).
                with torch.autocast(device_type=device_type):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                # gradient scaler for stability
                scaler = torch.GradScaler(device=device_type)

                # create scaled gradients of the loss
                scaler.scale(loss).backward()

                # unscales the gradients. call optimizer.step if gradient is not inf or nan
                scaler.step(optimizer)

                # updates the scale
                scaler.update()
            
            except:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
        if (epoch + 1) % 10 == 0: main_logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


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
      
        # execute early stopping
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
