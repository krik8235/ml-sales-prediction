import os
import sys
import json
import datetime
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

    # create X_test w/ column names for shap analysis and sensitive feature tracking
    X_test_with_col_names = X_test.copy()
    FEATURE_NAMES_PATH = os.path.join('preprocessors', 'feature_names.json')
    try:
        with open(FEATURE_NAMES_PATH, 'r') as f: feature_names = json.load(f)
    except FileNotFoundError: feature_names = X_test.columns.tolist()
    if len(X_test_with_col_names.columns) == len(feature_names): X_test_with_col_names.columns = feature_names

    # reconstruct optimal model
    MODEL_PATH = sys.argv[3]
    checkpoint = torch.load(MODEL_PATH)
    model = scripts.load_model(checkpoint=checkpoint)

    # perform inference
    y_pred, mse, exp_mae, rmsle = scripts.perform_inference(model=model, X=X_test, y=y_test, batch_size=checkpoint['batch_size'])

    # for fairness assessment - create df w/ y_pred, y_true, and sensitive features
    STOCKCODE = sys.argv[4]
    SENSITIVE_FEATURE = sys.argv[5]
    PRIVILEGED_GROUP = sys.argv[6]
    inference_df = pd.DataFrame(y_pred.cpu().numpy().flatten(), columns=['y_pred']) # type: ignore
    inference_df['y_true'] = y_test
    inference_df[SENSITIVE_FEATURE] = X_test_with_col_names[f'cat__{SENSITIVE_FEATURE}_{str(PRIVILEGED_GROUP)}'].astype(bool)
    inference_df.to_parquet(path=os.path.join('data', f'dfn_inference_results_{STOCKCODE}.parquet')) # dvc track

    # record inference metrics
    MODEL_INF_METRICS_PATH = os.path.join('metrics', f'dfn_inf_{STOCKCODE}.json')
    os.makedirs(os.path.dirname(MODEL_INF_METRICS_PATH), exist_ok=True)

    model_version = f"dfn_{STOCKCODE}_{os.getpid()}"
    inf_metrics = dict(
        stockcode=STOCKCODE,
        mse_inf=mse,
        mae_inf=exp_mae,
        rmsle_inf=rmsle,
        model_version=model_version,
        hparams=checkpoint['hparams'],
        optimizer=checkpoint['optimizer_name'],
        batch_size=checkpoint['batch_size'],
        lr=checkpoint['lr'],
        timestamp=datetime.datetime.now().isoformat()
    )
    with open(MODEL_INF_METRICS_PATH, 'w') as f: # dvc track
        json.dump(inf_metrics, f, indent=4)
        main_logger.info(f'... inference metrics saved to {MODEL_INF_METRICS_PATH} ...')


    ## shap analysis
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
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # shap raw data (dvc track)
    RAW_SHAP_OUT_PATH = os.path.join('reports', f'dfn_raw_shap_values_{STOCKCODE}.parquet')
    os.makedirs(os.path.dirname(RAW_SHAP_OUT_PATH), exist_ok=True)
    shap_df.to_parquet(RAW_SHAP_OUT_PATH, index=False)
    main_logger.info(f'... shap values saved to {RAW_SHAP_OUT_PATH} ...')

    # bar plot of mean abs shap vals (dvc report)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    shap_mean_abs_df = pd.DataFrame({'feature_name': feature_names, 'mean_abs_shap': mean_abs_shap.values })
    MEAN_ABS_SHAP_PATH = os.path.join('reports', f'dfn_shap_mean_abs_{STOCKCODE}.json')
    shap_mean_abs_df.to_json(MEAN_ABS_SHAP_PATH, orient='records', indent=4)
    main_logger.info(f'... mean abs shap values saved to {MEAN_ABS_SHAP_PATH} ...')

    # summary / beeswarm shap vals (dvc report)
    shap_melted_df = shap_df.melt(value_vars=feature_names, var_name='feature_name', value_name='shap_value') # type: ignore

    # add the corresponding original feature vals from x_test
    X_test_melted = X_test_with_col_names.melt(
        value_vars=X_test_with_col_names.columns, var_name='feature_name', value_name='feature_value' # type: ignore
    ).drop(columns=['feature_name']) # drop 'feature_name'

    shap_summary_df = pd.concat([shap_melted_df, X_test_melted], axis=1)

    SHAP_SUMMARY_PATH = os.path.join('reports', f'dfn_shap_summary_{STOCKCODE}.json')
    shap_summary_df.to_json(SHAP_SUMMARY_PATH, orient='records', indent=4)
    main_logger.info(f'... shap values summary saved to {SHAP_SUMMARY_PATH} ...')
