import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_log_error

from src._utils import main_logger


device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)


def make_prediction(X, y, model=None, criterion = nn.MSELoss()) -> tuple[np.ndarray, float, float, float]:
    """
    Makes a prediction from the dataset (X).
    Loads a model if file_path is given, else use a given model in the argument.
    Returns y_pred, loss, MSE for logged sales, MAE for actual quantity, and RMSLE for actual quantity.
    """
    from src.model.torch_model.scripts import create_torch_data_loader, load_model

    # load model
    model = model if model else load_model(input_dim=X.shape[1], model_name='dfn', trig='best')
    if model is None: raise Exception('No model found.')

    model.to(device)
    model.eval()

    # perform inference
    all_preds, all_targets = list(), list()
    running_loss = 0.0
    data_loader = create_torch_data_loader(X=X, y=y)

    with torch.inference_mode():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            loss = criterion(y_pred, target)
            running_loss += loss.item() * data.size(0)

            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # loss = criterion(y_tensor, y_pred)
    total_loss = running_loss / len(data_loader.dataset) # type: ignore

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # to avoid overflow, clip the values
    clipped_preds = np.clip(preds, -np.inf, 10)
    clipped_targets = np.clip(targets, -np.inf, 10)

    # computes performance metrics
    mse = mean_squared_error(targets, preds)
    exp_mae = mean_absolute_error(np.exp(clipped_targets), np.exp(clipped_preds))
    rmsle = root_mean_squared_log_error(np.exp(clipped_targets), np.exp(clipped_preds))

    main_logger.info(f"Predictions from loaded model\nLoss {total_loss:,.4f}, MSE for logged sales: {mse:,.4f}, MAE for actual quantity: {exp_mae:,.4f}, RMSLE for actual quantity: {rmsle:,.4f}\ny_pred:\n{y_pred}")

    return y_pred, mse, exp_mae, rmsle
