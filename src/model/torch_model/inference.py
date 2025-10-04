import os
import sys
import numpy as np
import pandas as pd
import torch
import shap

import src.model.torch_model.scripts as scripts
from src._utils import main_logger


if __name__ == '__main__':
    # handle device type
    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device_type == 'cpu':
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    # load test dataset
    X_TEST_PATH = sys.argv[1]
    Y_TEST_PATH = sys.argv[2]
    X_test, y_test = pd.read_parquet(X_TEST_PATH), pd.read_parquet(Y_TEST_PATH)

    # reconstruct optimal model
    MODEL_PATH = sys.argv[3]
    checkpoint = torch.load(MODEL_PATH)
    model = scripts.load_model(checkpoint=checkpoint)

    # perform inference
    y_pred, mse, exp_mae, rmsle = scripts.perform_inference(model=model, X=X_test, y=y_test, batch_size=checkpoint['batch_size'])

    # dvc track - y_pred
    STOCKCODE = sys.argv[4]
    y_pred_prediction_vals = pd.DataFrame(y_pred.cpu().numpy().flatten(), columns=['prediction']) # type: ignore
    y_pred_prediction_vals.to_parquet(path=os.path.join('data', f'dfn_inference_results_{STOCKCODE}.parquet'))

    # dvc track - metrics
    metrics = dict(mse=mse, mae=exp_mae, rmsle=rmsle)
    scripts.save_metrics(metrics=metrics, filepath=os.path.join('metrics', f'dfn_inference_results_{STOCKCODE}.json'))


    # compute shap vals
    model.eval()

    # prepare backgdound data
    X_test_tensor = torch.from_numpy(X_test.values.astype(np.float32)).to(device_type)

    # take the small samples from x_test as background
    background = X_test_tensor[np.random.choice(X_test_tensor.shape[0], 100, replace=False)].to(device_type)

    # define deepexplainer
    explainer = shap.DeepExplainer(model, background)

    # compute shap vals
    shap_values = explainer.shap_values(X_test_tensor) # outputs = numpy array or tensor

    # convert shap array to pandas df
    if isinstance(shap_values, list): shap_values = shap_values[0]
    if isinstance(shap_values, torch.Tensor): shap_values = shap_values.cpu().numpy()
    shap_values = shap_values.squeeze(axis=-1) # type: ignore
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns) # use the original feature names from the pandas df

    # dvc track - shap raw data to the dvc tracked path
    RAW_SHAP_OUT_PATH = os.path.join('reports', f'dfn_raw_shap_values_{STOCKCODE}.parquet')
    os.makedirs(os.path.dirname(RAW_SHAP_OUT_PATH), exist_ok=True)
    shap_df.to_parquet(RAW_SHAP_OUT_PATH, index=False)
    main_logger.info(f'... shap values saved to {RAW_SHAP_OUT_PATH} ...')

    # dvc track - bar plot mean abs shap vals
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    shap_mean_abs_df = pd.DataFrame({
        'feature_name': mean_abs_shap.index,
        'mean_abs_shap': mean_abs_shap.values
    })
    MEAN_ABS_SHAP_PATH = os.path.join('reports', f'dfn_shap_mean_abs_{STOCKCODE}.json')
    shap_mean_abs_df.to_json(MEAN_ABS_SHAP_PATH, orient='records', indent=4)
    main_logger.info(f'... mean abs shap values saved to {MEAN_ABS_SHAP_PATH} ...')

    # dvc track - plot summary / beeswarm shap vals
    shap_melted_df = shap_df.melt(
        value_vars=shap_df.columns, # type: ignore
        var_name='feature_name',
        value_name='shap_value'
    )

    # add the corresponding original feature vals from x_test
    X_test_melted = X_test.melt(
        value_vars=X_test.columns, # type: ignore
        var_name='feature_name',
        value_name='feature_value'
    ).drop(columns=['feature_name']) # drop 'feature_name'

    # combine shap values, feature names, and feature values
    shap_summary_df = pd.concat([shap_melted_df, X_test_melted], axis=1)

    SHAP_SUMMARY_PATH = os.path.join('reports', f'dfn_shap_summary_{STOCKCODE}.json')
    shap_summary_df.to_json(SHAP_SUMMARY_PATH, orient='records', indent=4)
    main_logger.info(f'... shap values summary saved to {SHAP_SUMMARY_PATH} ...')
