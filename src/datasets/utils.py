# Common utility functions for datasets
import torch
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from src.datasets.dataset import ChestXrayDataset
from src.datasets.dataloader import get_dataloader

def load_file_list(path):
    # Load a list of file names from a text file.
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def split_dataset(csv, val_size=0.15, test_size=0.15, random_state=42):
    df = pd.read_csv(csv)

    # Split dataset into training and validation sets.
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    # Adjust validation size relative to training set.
    val_size = val_size / (1 - test_size)
    # Split train_val_set into training and validation sets.
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state, shuffle=True)
    return train_df, val_df, test_df

def compute_mean_std(df, img_dir, transform=None, class_names=None):
    # Compute the mean and standard deviation of a dataset.
    if transform is None:
        transform = transforms.ToTensor()
    
    dataset = ChestXrayDataset(df, img_dir, transform=transform, class_names=class_names)
    loader = get_dataloader(dataset)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.numpy(), std.numpy()