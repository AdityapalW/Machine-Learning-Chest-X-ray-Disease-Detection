# Define how images and labels are loaded from the Kaggle dataset CSVs.

'''
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, class_names=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image names and labels
            img_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied
            class_names (list): List of disease class names (for column indexing)
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names or [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        # Labels → tensor of 14 binary values
        labels = self.data[self.class_names].iloc[idx].values.astype("float32")

        if self.transform:
            image = self.transform(image)

        return image, labels


def get_transforms(train=True):
    """Define train vs. validation/test transforms"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(csv_train, csv_val, img_dir, batch_size=32, num_workers=2):
    """Return PyTorch DataLoader objects for train/val"""
    train_dataset = ChestXrayDataset(
        csv_file=csv_train,
        img_dir=img_dir,
        transform=get_transforms(train=True)
    )

    val_dataset = ChestXrayDataset(
        csv_file=csv_val,
        img_dir=img_dir,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
'''