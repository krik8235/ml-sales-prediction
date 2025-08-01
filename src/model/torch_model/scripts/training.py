import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src._utils import main_logger


def train_model(
        X_train, X_val, y_train, y_val, 
        model,
        batch_size,
        optimizer,
        criterion = nn.MSELoss(),
        num_epochs: int = 50,
        min_delta: float = 1e-5,
        patience: int = 10
    ) -> tuple[nn.Module, float]:

    from src.model.torch_model.scripts.tuning import create_torch_data_loader
    
    X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
        X_train, y_train, test_size=5000, shuffle=True, random_state=42
    )
    train_data_loader = create_torch_data_loader(X=X_train_search, y=y_train_search, batch_size=batch_size)
    val_data_loader = create_torch_data_loader(X=X_val_search, y=y_val_search, batch_size=batch_size)

    train_data_loader = create_torch_data_loader(X_train, y_train, batch_size=batch_size)
    val_data_loader = create_torch_data_loader(X=X_val, y=y_val, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # start training and validation
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0: main_logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_data_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                outputs_val = model(batch_X_val)
                val_loss += criterion(outputs_val, batch_y_val).item()

        val_loss /= len(val_data_loader)
      
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                main_logger.info(f'Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.')
                break

    return model, best_val_loss

