import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_log_error

from src._utils import main_logger



def make_prediction(model, X, y) -> tuple[np.ndarray, float, float, float]:
    """Makes a prediction, evaluates the performance, and returns the evaluation metrics."""

    y_pred = model.predict(X)

    # to avoid overflow, clip the values
    clipped_preds = np.clip(y_pred, -np.inf, 10)
    clipped_targets = np.clip(y, -np.inf, 10)

    # computes performance metrics
    mse = mean_squared_error(y, y_pred)
    exp_mae = mean_absolute_error(np.exp(clipped_targets), np.exp(clipped_preds))
    rmsle = root_mean_squared_log_error(np.exp(clipped_targets), np.exp(clipped_preds))

    main_logger.info(f'MSE for logged sales: {mse:,.4f}, MAE for actual quanity: {exp_mae:,.4f} units, RMSLE for actual quanity: {rmsle:,.4f}')

    return y_pred, mse, exp_mae, rmsle
