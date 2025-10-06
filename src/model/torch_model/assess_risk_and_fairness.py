import os
import json
import datetime
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_log_error

from src._utils import main_logger


def calculate_fairness_metrics(
        df: pd.DataFrame,
        sensitive_feature_col: str,
        label_col: str = 'y_true',
        prediction_col: str = 'y_pred',
        privileged_group: int = 1,
        mod_threshold: float = 0.1,
    ) -> dict:

    metrics = dict()
    unprivileged_group = 0 if privileged_group == 1 else 1

    ## 1. risk assessment - predictive performance metrics by group
    for group, name in zip([unprivileged_group, privileged_group], ['unprivileged', 'privileged']):
        subset = df[df[sensitive_feature_col] == group]
        if len(subset) == 0: continue

        y_true = subset[label_col].values
        y_pred = subset[prediction_col].values

        metrics[f'mse_{name}'] = float(mean_squared_error(y_true, y_pred)) # type: ignore
        metrics[f'mae_{name}'] = float(mean_absolute_error(y_true, y_pred)) # type: ignore
        metrics[f'rmsle_{name}'] = float(root_mean_squared_log_error(y_true, y_pred)) # type: ignore

        # mean prediction (outcome disparity component)
        metrics[f'mean_prediction_{name}'] = float(y_pred.mean()) # type: ignore

    ## 2. bias assessment - fairness metrics
    # absolute mean error difference
    mae_diff = metrics.get('mae_unprivileged', 0) - metrics.get('mae_privileged', 0)
    metrics['mae_diff'] = float(mae_diff)

    # mean outcome difference
    mod = metrics.get('mean_prediction_unprivileged', 0) - metrics.get('mean_prediction_privileged', 0)
    metrics['mean_outcome_difference'] = float(mod)
    metrics['is_mod_acceptable'] = 1 if abs(mod) <= mod_threshold else 0

    return metrics


def main():
    parser = argparse.ArgumentParser(description='assess bias and fairness metrics on model inference results.')
    parser.add_argument('inference_file_path', type=str, help='parquet file path to the inference results w/ y_true, y_pred, and sensitive feature cols.')
    parser.add_argument('metrics_output_path', type=str, help='json file path to save the metrics output.')
    parser.add_argument('sensitive_feature_col', type=str, help='column name of sensitive features')
    parser.add_argument('stockcode', type=str)
    parser.add_argument('privileged_group', type=int, default=1)
    parser.add_argument('mod_threshold', type=float, default=.1)
    args = parser.parse_args()

    try:
        # load inf df
        df_inference = pd.read_parquet(args.inference_file_path)
        LABEL_COL = 'y_true'
        PREDICTION_COL = 'y_pred'
        SENSITIVE_COL = args.sensitive_feature_col

        # check if df_inference contains cols
        if SENSITIVE_COL not in df_inference.columns:
            main_logger.error(f'... {SENSITIVE_COL} not in the inf df. skip operation ...')
            exit(1)

        if LABEL_COL not in df_inference.columns or PREDICTION_COL not in df_inference.columns:
            main_logger.error(f'... {LABEL_COL} or {PREDICTION_COL} not in the inf df. skip operation ...')
            exit(1)

        # compute fairness metrics
        metrics = calculate_fairness_metrics(
            df=df_inference,
            sensitive_feature_col=SENSITIVE_COL,
            label_col=LABEL_COL,
            prediction_col=PREDICTION_COL,
            privileged_group=args.privileged_group,
            mod_threshold=args.mod_threshold,
        )

        # add items to metrics
        metrics['model_version'] = f'dfn_{args.stockcode}_{os.getpid()}'
        metrics['sensitive_feature'] = args.sensitive_feature_col
        metrics['privileged_group'] = args.privileged_group
        metrics['mod_threshold'] = args.mod_threshold
        metrics['stockcode'] = args.stockcode
        metrics['timestamp'] = datetime.datetime.now().isoformat()

        # load metrics (dvc track)
        with open(args.metrics_output_path, 'w') as f:
            json_metrics = { k: (v if pd.notna(v) else None) for k, v in metrics.items() }
            json.dump(json_metrics, f, indent=4)
            main_logger.info(f'... successfully saved risk and fairness metrics to {args.metrics_output_path} ...')

    except Exception as e:
        main_logger.error(f'... an error occurred during risk and fairness assessment: {e} ...')
        exit(1)

if __name__ == '__main__':
    main()
