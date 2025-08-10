import os
import datetime

current_dir =  os.getcwd()
MODEL_SAVE_PATH = os.path.join(current_dir, 'models')


def create_file_path(model_name: str = 'gdm', version_tag = None, trig: str = 'best') -> tuple[str, str]:
    """Creates and returns a file path and a file name (timestamped) for model serialization."""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # create a folder path
    folder_path = os.path.join(MODEL_SAVE_PATH, f'{model_name}_{trig}')
    os.makedirs(folder_path, exist_ok=True)

    # create a file path
    file_ext = 'pth' if model_name == 'dfn' else 'pkl'
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f'{model_name}_{timestamp}{f"_{version_tag}" if version_tag else ""}.{file_ext}'
    file_path = os.path.join(folder_path, file_name)

    return file_path, file_name


def retrieve_file_path(folder_path: str) -> tuple[str, str]:
    os.makedirs(folder_path, exist_ok=True)
    all_entries = os.listdir(folder_path)
    files_in_folder = [f for f in all_entries if os.path.isfile(os.path.join(folder_path, f))]

    file_path, file_name = '', ''
    if files_in_folder:
        files_in_folder.sort()
        file_name = files_in_folder[-1]
        file_path = os.path.join(folder_path, file_name)

    return  file_path, file_name
