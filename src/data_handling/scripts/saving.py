import os
import pandas as pd

os.makedirs('data', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)


def save_df_to_csv(df: pd.DataFrame) -> tuple[str, str]: # type: ignore
    """save dataframe to csv file and returns file_path and file name"""

    if df is None: raise Exception('Dataframe is missing')
    
    current_dir =  os.getcwd()
    file_name = 'processed_df.csv'
    folder_path = os.path.join(current_dir, 'data', 'processed')
    file_path = os.path.join(folder_path, file_name)

    df.to_csv(file_path)

    return file_path, file_name
