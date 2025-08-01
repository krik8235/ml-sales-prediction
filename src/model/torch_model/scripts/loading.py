import os
import torch
import torch.nn as nn

from src.model.torch_model.scripts.pretrained_base import DFN
from src._utils import main_logger, MODEL_SAVE_PATH, retrieve_file_path


def construct_model_from_state(input_dim: int, state: dict):
    """Reconstruct a torch model from a state """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =  DFN(input_dim=input_dim)

    try:
        model.to(device=device)
        model.load_state_dict(state)
    except:
        main_logger.error('Failed to laod model')
    
    return model


def load_model(input_dim=None, model_name: str = 'dfn', trig: str ='best') -> nn.Module:
    """
    Loads and returns the latest tensor model with a state dictionary for reconstruction.
    When failed to load the model, returns a new untrained model with a potentially emplty state dictionary.    
    """
    try:
        # initialize model and state_dict
        model = DFN(input_dim=input_dim)
        state_dict = model.state_dict()
    
        # load saved state dict (best performing model)
        folder_path = os.path.join(MODEL_SAVE_PATH, f'{model_name}_{trig}')
        file_path, _ = retrieve_file_path(folder_path=folder_path)
        if not file_path: raise Exception('File path not found.')
        saved_state_dict = torch.load(file_path)

        # create pretrained state dict and load it to the init model
        pretrained_dict = {k: v for k, v in saved_state_dict.items() if k in state_dict and v.shape == state_dict[k].shape}
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)

        return model
    
    except Exception as e:
        main_logger.error(f"failed to load PyTorch model for retraining: {e}. Returning new untrained model with state_dict.")
        raise Exception('failed to load the model.')
