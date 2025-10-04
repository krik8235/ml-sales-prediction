import os
import uuid
import json
import datetime
import torch

from src._utils import main_logger, create_file_path, s3_upload


def save_model_to_local(checkpoint: dict, model_name: str = 'dfn', trig: str = 'best'):
    file_path, _ = create_file_path(model_name=model_name, trig=trig)

    torch.save(checkpoint, file_path)
    main_logger.info(f"pytorch model saved to {file_path}")
    return file_path



def save_metrics(metrics: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        metrics['timestamp'] = datetime.datetime.now().isoformat()
        json.dump(metrics, f, indent=4) # dvc track

    main_logger.info(f'... metrics saved to {filepath} ...')



def save_historical_metric_to_s3(stockcode: str, metrics: dict, model_version: str, **kwargs):
    record = {
        **metrics, **kwargs, # add config from checkpoint
        'stockcode': stockcode,
        'model_version': model_version,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    temp_dir = os.path.join('tmp', 'metrics_history')
    os.makedirs(temp_dir, exist_ok=True)

    # use a unique filename based on time and a uuid for safety
    filename = f'dfn_{stockcode}_{int(datetime.datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}.json'
    local_filepath = os.path.join(temp_dir, filename)

    # save record in json file
    with open(local_filepath, 'w') as f:
        json.dump(record, f, indent=2)
        main_logger.info(f'... historical metric saved locally to {local_filepath} ...')

    # s3 upload
    s3_key = os.path.join('historical_metrics', filename)
    s3_upload(file_path=local_filepath, s3_key=s3_key)
    main_logger.info(f'... historical metric uploaded to s3 {s3_key} ...')

    # clean up local file
    os.remove(local_filepath)
