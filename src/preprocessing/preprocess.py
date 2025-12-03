# Define how images and labels are loaded from the Kaggle dataset CSVs.

from torchvision import transforms

# for milestone 1: baseline model for binary classification
# class_name = ["Disease"] 
# replace all disease labels in csv file to this

def get_transforms(mean, std, train=True):
    # Transform images in training dataset.
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # Add augmentation to prevent overfitting.
            transforms.RandomRotation(10),
             transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            ),
            transforms.ToTensor(),  # Convert PIL image to PyTorch tensor.
            transforms.Normalize(mean=mean, std=std)  # Normalize for ImageNet pre-training data.
        ])
    # Transform images in validation/test dataset.
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])