# Define how images and labels are loaded from the Kaggle dataset CSVs.

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform, class_names):
        self.data = pd.read_csv(csv_file)  # Load CSV with image names and labels.
        self.img_dir = img_dir  # Directory containing images  
        self.transform = transform  # Image transformations
        self.class_names = class_names

        # Convert multi-label strings into binary columns for each class.
        # E.g., "Atelectasis|Effusion" -> [1, 0, 1, 0, 0, 0, ..., 0]
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