# Common utility functions for datasets
import pandas as pd
from sklearn.model_selection import train_test_split

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