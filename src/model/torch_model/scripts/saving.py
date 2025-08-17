import torch

from src._utils import main_logger, create_file_path


def save_model_to_local(checkpoint: dict, model_name: str = 'dfn', trig: str = 'best'):
    file_path, _ = create_file_path(model_name=model_name, trig=trig)

    torch.save(checkpoint, file_path)
    main_logger.info(f"pytorch model saved to {file_path}")
    return file_path
