import os
import torch
import numpy as np

import src.data_handling as data_handling
import src.model.torch_model.scripts as scripts
from src._utils import s3_upload, main_logger


def retrain():
    X_train, X_val, X_test, y_train, y_val, y_test, _ = data_handling.main_script()

    batch_size = 32
    train_data_loader = scripts.create_torch_data_loader(X=X_train, y=y_train, batch_size=batch_size)
    val_data_loader = scripts.create_torch_data_loader(X=X_val, y=y_val, batch_size=batch_size)

    file_path = os.path.join('models', 'production', 'dfn_best.pth')
    model = scripts.load_model(input_dim=X_train.shape[1], file_path=file_path)

    model, _ = scripts.train_model(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
        criterion=torch.nn.MSELoss(),
        num_epochs=50,
        min_delta=0.00001,
        patience=10,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        device_type='cpu'
    )

    # X = np.concatenate([X_train, X_test], axis=0) if isinstance(X_train, np.ndarray) else pd.concat([X_train, X_test], ignore_index=True)
    input_tensor = torch.tensor(X_test, dtype=torch.float32)

    model.eval()
    epsilon = 1e-5
    with torch.inference_mode():
        y_pred = model(input_tensor)
        y_pred = y_pred.cpu().numpy().flatten()
        y_pred_actual = np.exp(y_pred + epsilon)
        main_logger.info(f"retrained primary model's prediction - actual quantity (units) {y_pred_actual}")
    return model


if __name__ == '__main__':
    PRODUCTION_MODEL_FOLDER_PATH = 'models/production'
    DFN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'dfn_best.pth')
    model = retrain()
    torch.save(model.state_dict(), DFN_FILE_PATH)
    s3_upload(file_path=DFN_FILE_PATH)
