# Define how images and labels are loaded from the Kaggle dataset CSVs.

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform, class_names, file_list=None):
        """
        Args:
            csv_file: Path to the CSV file with image labels (e.g., DATA_ENTRY_2017.CSV).
            img_dir: Directory with all the images.
            transform: Transformations to apply to the images.
            class_names: List of class names for multi-label classification.
            file_list: Path to a text file containing the list of image filenames to include.
        """
        self.data = pd.read_csv(csv_file)  # Load the CSV file
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names

        # Filter the data based on the file list (if provided)
        if file_list is not None:
            if isinstance(file_list, str):
                # It's a path to a .TXT file
                with open(file_list, "r") as f:
                    valid_files = set(f.read().splitlines())
            elif isinstance(file_list, list):
                # It's already a Python list (after splitting)
                valid_files = set(file_list)
            else:
                raise ValueError("file_list must be a list of filenames or a path to a file")
            
            self.data = self.data[self.data["Image Index"].isin(valid_files)]

        # Convert multi-label strings into binary columns for each class
        for label in self.class_names:
            self.data[label] = self.data["Finding Labels"].apply(lambda x: 1.0 if label in x else 0.0)
    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # Fetch one image with image name from CSV.
        image = Image.open(img_name).convert("RGB")  # Convert to RGB for ImageNet compatibility.

        #################################
        labels = self.data[self.class_names].iloc[idx].values.astype("float32")
        #################################

        if self.transform:
            image = self.transform(image)

        return image, labels