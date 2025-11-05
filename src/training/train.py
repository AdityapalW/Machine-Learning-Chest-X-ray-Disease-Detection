import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.models.baseline_cnn import BaselineCNN

# def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device):
#     '''
#     for epoch in range(num_epochs):
#         model.train()
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss_value = loss_fn(outputs, labels)
#             loss_value.backward()
#             optimizer.step()
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value.item()}")
#     '''
    
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with Image Index and Finding Labels")
    p.add_argument("--images-dir", required=True, help="directory with image files")
    p.add_argument("--output-dir", default="outputs/baseline")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="use mixed precision")
    return p.parse_args()

def multi_hot_to_binary(labels_tensor):
    if labels_tensor.dim() == 1:
        # already scalar (edge case)
        return (labels_tensor > 0).long()
    return (labels_tensor.sum(dim=1) > 0).long()

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    for imgs, labels_multi in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        labels_multi = labels_multi.to(device).float()  # Ensure labels are on the same device and in float format
        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):  # Updated for deprecation warning
            logits = model(imgs)  # Model output (logits)
            loss = criterion(logits, labels_multi)  # Compute loss
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