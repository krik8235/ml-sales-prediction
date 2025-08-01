import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_log_error

from src._utils import main_logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def load_model(X = None, folder_path: str = None, file_path: str = None) -> tuple[nn.Module, dict]: # type: ignore
#     """Loads the latest models from the given folder"""

#     if not file_path:
#         files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#         if not files: raise Exception(f"No files found in the specified folder: {folder_path}")
#         files.sort()
#         file_path = files[-1]

#     if X is not None:
#         model = DFN(input_dim=X.shape[1])
#         try:
#             state_dict = torch.load(file_path, map_location=device)
#             model.load_state_dict(state_dict)
#         except Exception as e: raise Exception(f"Error loading model from {file_path}: {e}")

#         model.to(device)
#         model.eval()
#         return model, state_dict
    
#     else:
#         state_dict = torch.load(file_path, map_location=device)
#         return None, state_dict # type: ignore


def make_prediction(X, y, model=None, criterion = nn.MSELoss()) -> tuple[np.ndarray, float, float, float]:
    """
    Makes a prediction from the dataset (X). 
    Loads a model if file_path is given, else use a given model in the argument.
    Returns y_pred, loss, MSE for logged sales, MAE for actual sales, and RMSLE for actual sales.
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

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            loss = criterion(y_pred, target)
            running_loss += loss.item() * data.size(0)

            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())


                # if y_pred.dim() > y_tensor.dim() and y_pred.shape[1] == 1: y_pred = y_pred.squeeze(1)
                # elif y_tensor.dim() > y_pred.dim() and y_tensor.shape[1] == 1: y_tensor = y_tensor.squeeze(1)
                # elif y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)
                # elif y_tensor.dim() == 1: y_tensor = y_tensor.unsqueeze(1)
                
                # if y_pred.shape != y_tensor.shape:
                #     main_logger.error(f"Warning: y_pred shape {y_pred.shape} and y_tensor shape {y_tensor.shape} do not match.")
                #     pass 
        # except:
        #     y_pred = model(X_tensor).cpu().numpy()
    
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

    main_logger.info(f"Predictions from loaded model\nLoss {total_loss:,.4f}, MSE for logged sales: {mse:,.4f}, MAE for actual sales: {exp_mae:,.4f}, RMSLE for actual sales: {rmsle:,.4f}\ny_pred:\n{y_pred}")

    return y_pred, mse, exp_mae, rmsle
