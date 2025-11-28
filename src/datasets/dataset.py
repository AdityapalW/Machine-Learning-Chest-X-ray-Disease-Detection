# Define how images and labels are loaded from the Kaggle dataset CSVs.

import os
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform, class_names):
        """
        Args:
            df: DataFrame containing image filenames and labels.
            img_dir: Directory with all the images.
            transform: Transformations to apply to the images.
            class_names: List of class names for multi-label classification.
            file_list: Path to a text file containing the list of image filenames to include.
        """
        self.data = df.reset_index(drop=True)  # Reset the splitted DataFrame index to start from 0.
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names

        # Convert multi-label strings into binary columns for each class
        for label in self.class_names:
            self.data[label] = self.data["Finding Labels"].apply(lambda x: 1.0 if label in x else 0.0)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # Fetch the image's name of index idx from CSV.
        image = Image.open(img_name).convert("RGB")  # Convert to RGB for ImageNet compatibility.

        # Extract the 14*1 label array for each image, and convert labels to float32.
        labels = self.data[self.class_names].iloc[idx].values.astype("float32")

        if self.transform:
            image = self.transform(image)

        return image, labels