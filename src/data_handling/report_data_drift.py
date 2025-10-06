import os
import sys
import json
import pandas as pd
import datetime
from dotenv import load_dotenv

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import CloudWorkspace

import src.data_handling.scripts as scripts
from src._utils import main_logger


if __name__ == '__main__':
    # initiate evently cloud workspace
    load_dotenv(override=True)
    ws = CloudWorkspace(token=os.getenv('EVENTLY_API_TOKEN'), url='https://app.evidently.cloud')

    # retrieve evently project
    project = ws.get_project('01998e47-e3d7-7ee7-ae5e-b6fdf3f80fff')

    # retrieve paths from the command line args
    REFERENCE_DATA_PATH = sys.argv[1]
    CURRENT_DATA_PATH = sys.argv[2]
    REPORT_OUTPUT_PATH = sys.argv[3]
    METRICS_OUTPUT_PATH = sys.argv[4]
    STOCKCODE = sys.argv[5]

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)

    # extract datasets
    reference_data_full = pd.read_csv(REFERENCE_DATA_PATH)
    reference_data_stockcode = reference_data_full[reference_data_full['stockcode'] == STOCKCODE]
    current_data_stockcode = pd.read_parquet(CURRENT_DATA_PATH)

    # define data schema
    nums, cats = scripts.categorize_num_cat_cols(df=reference_data_stockcode)
    for col in nums: current_data_stockcode[col] = pd.to_numeric(current_data_stockcode[col], errors='coerce')

    schema = DataDefinition(numerical_columns=nums, categorical_columns=cats)

    # define evently dataset w/ the data schema
    eval_data_1 = Dataset.from_pandas(reference_data_stockcode, data_definition=schema)
    eval_data_2 = Dataset.from_pandas(current_data_stockcode, data_definition=schema)

    # run
    report = Report(metrics=[DataDriftPreset()])
    data_eval = report.run(reference_data=eval_data_1, current_data=eval_data_2)
    data_eval.save_html(REPORT_OUTPUT_PATH)
    main_logger.info(f'... drift report saved to {REPORT_OUTPUT_PATH} ...')


    # extract metrics from the report
    report_dict = json.loads(data_eval.json())
    num_drifts = report_dict['metrics'][0]['value']['count']
    shared_drifts = report_dict['metrics'][0]['value']['share']
    metrics = dict(
        drift_detected=bool(num_drifts > 0.0), num_drifts=num_drifts, shared_drifts=shared_drifts,
        num_cols=nums,
        cat_cols=cats,
        stockcode=STOCKCODE,
        timestamp=datetime.datetime.now().isoformat(),
    )

    # store the metrics in json for dvc to track
    with open(METRICS_OUTPUT_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        main_logger.info(f'... drift metrics saved to {METRICS_OUTPUT_PATH}... ')


    # add eval to the project
    if project is not None: ws.add_run(project_id=project.id, run=data_eval, include_data=False, name=STOCKCODE)

    # raise an error if drift is detected and stop the dvc pipeline
    if num_drifts > 0.0: sys.exit('❌ FATAL: data drift detected. stopping pipeline')
