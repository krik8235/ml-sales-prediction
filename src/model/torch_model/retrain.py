import os
import torch
import numpy as np
import pandas as pd

import src.data_handling as data_handling
import src.model.torch_model.scripts as scripts
from src._utils import s3_upload, main_logger


def retrain():
    # device
    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)

    # load checkpoint, model, optimizer
    file_path = os.path.join('models', 'production', 'dfn_best.pth')
    model, optimizer, checkpoint = scripts.load_model_and_optimizer(file_path=file_path, device_type=device_type)

    # prep for training data
    batch_size = checkpoint['batch_size']
    X_train, X_val, _, y_train, y_val, _, _ = data_handling.main_script()
    train_data_loader = scripts.create_torch_data_loader(X=X_train, y=y_train, batch_size=batch_size)
    val_data_loader = scripts.create_torch_data_loader(X=X_val, y=y_val, batch_size=batch_size)

    # train the model
    model, _ = scripts.train_model(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        num_epochs=500,
        min_delta=1e-5,
        patience=10,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        batch_size=batch_size,
        device_type=device_type,
    )

    # perform inf
    X = np.concatenate([X_train, X_val], axis=0) if isinstance(X_train, np.ndarray) else pd.concat([X_train, X_val], ignore_index=True) # type: ignore
    input_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    # input_tensor = input_tensor.to(device)

    model.eval()
    epsilon = 0
    with torch.inference_mode():
        y_pred = model(input_tensor)
        y_pred = y_pred.cpu().numpy().flatten()
        y_pred_actual = np.exp(y_pred + epsilon)
        main_logger.info(f"retrained primary model's prediction - actual quantity (units) {y_pred_actual}")

    # update checkpoint
    checkpoint['state_dict'] = model.state_dict()
    checkpoint['input_dim'] = X_train.shape[1]

    return checkpoint


if __name__ == '__main__':
    PRODUCTION_MODEL_FOLDER_PATH = 'models/production'
    DFN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'dfn_best.pth')
    checkpoint = retrain()
    torch.save(checkpoint, DFN_FILE_PATH)
    s3_upload(file_path=DFN_FILE_PATH)
