# Define how images and labels are loaded from the Kaggle dataset CSVs.

from torchvision import transforms

def get_transforms(train=True):
    # Transform images in training dataset.
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # Add augmentation to prevent overfitting.
            transforms.RandomRotation(10),
            transforms.ToTensor(),  # Convert PIL image to PyTorch tensor.
            transforms.Normalize(  # Normalize for ImageNet pre-training data.
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    # Transform images in validation/test dataset.
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
'''
def get_dataloader(data_csv, img_dir, batch_size=32, num_workers=2):
    """Return PyTorch DataLoader objects for training dataset."""
    train_dataset = ChestXrayDataset(
        csv_file=data_csv,
        img_dir=img_dir,
        transform=get_transforms(train=True)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
'''