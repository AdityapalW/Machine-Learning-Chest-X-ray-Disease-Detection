import torch
from torch.utils.data import DataLoader
from src.datasets.dataset import ChestXrayDataset
from src.datasets.dataloader import get_dataloader
from src.preprocessing.preprocess import get_transforms
from src.models.network import get_model
from src.training.loss import get_loss_function
from src.training.train import train_model

def main():
    train_csv = "data/raw/DATA_ENTRY_2017.CSV"  # CSV file with image labels
    train_val_list = "data/raw/TRAIN_VAL_LIST_NIH.TXT"  # File list for training/validation
    test_list = "data/raw/TEST_LIST_NIH.TXT"  # File list for testing
    bbox_csv = "data/raw/BBOX_LIST_2017_OFFICIAL_NIH.CSV"  # Bounding box annotations (not used in main)
    img_dir = "data/raw/images-224/images-224/"  # Directory containing the images
    num_epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    lr = 1e-4  # Learning rate for optimizer

    # Get transforms for training and validation
    train_transforms = get_transforms(train=True)  # Applies augmentation for training
    val_transforms = get_transforms(train=False)  # No augmentation for validation

    # Class names for multi-label classification
    class_names = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]

    # Create datasets, filtering by file lists
    train_dataset = ChestXrayDataset(train_csv, img_dir, train_transforms, class_names, file_list=train_val_list)
    val_dataset = ChestXrayDataset(train_csv, img_dir, val_transforms, class_names, file_list=test_list)

    # Create DataLoaders for batching
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=2)  # Shuffles and batches training data
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, num_workers=2)  # Batches validation data without shuffling

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # Get model, loss function, and optimizer
    model = get_model()  # Returns the model (e.g., BaselineCNN)
    model = model.to(device)
    loss_fn = get_loss_function()  # Returns BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    # Train the model
    train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device)  # Trains for num_epochs

    print("Training complete!")

if __name__ == "__main__":
    main()