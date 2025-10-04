import os
import re
import torch
import torch.nn as nn

from src.model.torch_model.scripts.pretrained_base import DFN
from src.model.torch_model.scripts.tuning import handle_optimizer
from src._utils import main_logger, MODEL_SAVE_PATH, retrieve_file_path


device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)


# for inference
def load_model(checkpoint: dict = {}, model_name: str = 'dfn', trig: str ='best', file_path=None) -> nn.Module:
    """
    Loads and reconstructs the best performing model from the tuning result:
    checkpoint = {
        'state_dict': best_model.state_dict(),
        'hparams': best_hparams,
        'input_dim': X_train.shape[1],
        'optimizer_name': optimizer_name,
        'lr': lr,
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_size': batch_size
    }
    """
    try:
        if not checkpoint:
            if not file_path:
                folder_path = os.path.join(MODEL_SAVE_PATH, f'{model_name}_{trig}')
                file_path, _ = retrieve_file_path(folder_path=folder_path)
                if not file_path: raise Exception('file path not found.')

            checkpoint = torch.load(file_path, weights_only=False, map_location=device)

        state_dict = checkpoint['state_dict']
        input_dim = checkpoint['input_dim']
        hparams = checkpoint['hparams']
        num_layers = hparams['num_layers']
        hidden_units_per_layer = [v for k, v in hparams.items() if 'n_units_layer_' in k]
        if not hidden_units_per_layer: # construct from state_dict
            hidden_units_per_layer = []
            layer_indices = sorted(
                [int(re.search(r'\.(\d+)\.weight', k).group(1)) for k in state_dict.keys() if re.search(r'\.weight$', k)] # type: ignore
            )
            for index in layer_indices[:-1]: hidden_units_per_layer.append(state_dict[f'model_layers.{index}.weight'].shape[0])

        dropout_rates = [v for k, v in hparams.items() if 'dropout_rate_layer_' in k]
        if not dropout_rates: dropout_rates = [0.1] * num_layers

        model = DFN(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_units_per_layer=hidden_units_per_layer,
            batch_norm=hparams.get('batch_norm', False),
            dropout_rates=dropout_rates
        )
        model.load_state_dict(state_dict, strict=True)

        main_logger.info(f'model loaded and reconstructed using the best hparams: {hparams}')
        return model

    except Exception as e:
        main_logger.error(f"... failed to load pytorch model: {e}")
        raise


# for training
def load_model_and_optimizer(checkpoint: dict = {}, model_name: str = 'dfn', trig: str ='best', file_path=None, device_type=None):
    try:
        # device
        device_type = device_type if device_type else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        device = torch.device(device_type)

        # model
        model = load_model(checkpoint=checkpoint, file_path=file_path)
        model.to(device)

        # optimizer
        if not checkpoint:
            if not file_path:
                folder_path = os.path.join(MODEL_SAVE_PATH, f'{model_name}_{trig}')
                file_path, _ = retrieve_file_path(folder_path=folder_path)
                if not file_path: raise Exception('file path not found.')

            checkpoint = torch.load(file_path, weights_only=False, map_location=device)

        optimizer_name = checkpoint['optimizer_name']
        lr = checkpoint['lr']
        optimizer = handle_optimizer(optimizer_name=optimizer_name, model=model, lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer, checkpoint

    except Exception as e:
        main_logger.error(f"... failed to load the model or optimizer: {e}")
        raise
