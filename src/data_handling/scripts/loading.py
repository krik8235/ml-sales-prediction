import os
import datetime
import pandas as pd

from src._utils import retrieve_file_path


def sanitize_column_names(df):
    for col in df.columns:
        col_revised = col.replace(' ', '_').lower().replace('__', '_')
        df[col_revised] = df[col]
        df = df.drop(columns=col, axis='columns')
    return df


def load_original_dataframe() -> pd.DataFrame: # type: ignore
    """Downloads data from CSV file."""

    current_dir =  os.getcwd()
    file_name = 'online_retail.csv'
    file_path = os.path.join(current_dir, 'data', 'raw', file_name)
    df = pd.read_csv(file_path)
    df = sanitize_column_names(df=df)
    return df


def load_post_feature_engineer_dataframe() -> tuple[pd.DataFrame, str , str]:
    current_dir =  os.getcwd()
    folder_path = os.path.join(current_dir, 'data', 'processed')
    file_path, file_name = retrieve_file_path(folder_path=folder_path)

    if not file_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f'processed_df_{timestamp}.csv'
        file_path = os.path.join(current_dir, 'data', 'processed', file_name)

    try:
        df = pd.read_csv(file_path)
        if 'Unnamed: 0' in df.columns: df = df.drop('Unnamed: 0', axis=1)
        return df, file_path, file_name

    except:
        raise Exception('Failed to load dataframe.')
