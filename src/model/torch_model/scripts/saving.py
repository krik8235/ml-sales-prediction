import os
import torch
import joblib

from src._utils import main_logger, create_file_path, s3_upload


def save_model(model, model_name: str = 'dfn', trig: str = 'best'):
    """Saves trained DFNs to local and S3 bucket."""

    file_path, file_name = create_file_path(model_name=model_name, trig=trig)

    torch.save(model.state_dict(), file_path)
    main_logger.info(f"PyTorch model saved to {file_path}")

    if trig == 'best':
        joblib.dump(model.state_dict(), os.path.join(file_path, file_name))
        s3_upload(file_path=file_path, file_name=file_name)

    return file_path
