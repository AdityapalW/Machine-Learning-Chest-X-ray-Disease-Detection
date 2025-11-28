from tqdm import tqdm
import torch

def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, scaler=None):
    """
    Trains the model for a specified number of epochs and validates it after each epoch.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        loss_fn: Loss function to use.
        optimizer: Optimizer for updating model parameters.
        num_epochs: Number of epochs to train for.
        device: Device to train on (e.g., 'cuda' or 'cpu').
        scaler: Optional GradScaler for mixed precision training.

    Returns:
        A dictionary containing training and validation losses for each epoch.
    """
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        history["train_loss"].append(train_loss)

        # Validate the model
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels_multi in val_loader:
                imgs = imgs.to(device)
                labels_multi = labels_multi.to(device).float()  # Ensure labels are on the same device and in float format
                logits = model(imgs)
                loss = loss_fn(logits, labels_multi)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return history

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    for imgs, labels_multi in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        labels_multi = labels_multi.to(device).float()  # Ensure labels are on the same device and in float format.

        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):  # Updated for deprecation warning.
            logits = model(imgs)  # Raw model output for one batch
            loss = criterion(logits, labels_multi)  # Compute loss.
        
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)