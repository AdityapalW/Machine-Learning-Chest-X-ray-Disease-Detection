import torch
from src.datasets.dataset import ChestXrayDataset
from src.datasets.dataloader import get_dataloader
from src.datasets.utils import split_dataset
from src.preprocessing.preprocess import get_transforms
from models.custom_cnn import CustomCNN
from src.training.loss import get_loss_function
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model

def main():
    data_csv = "data/raw/sample_labels.CSV"  # CSV file with image labels
    '''
    train_val_files = "data/raw/TRAIN_VAL_LIST_NIH.TXT"  # File list for training/validation
    test_files = "data/raw/TEST_LIST_NIH.TXT"  # File list for testing
    bbox_csv = "data/raw/BBOX_LIST_2017_OFFICIAL_NIH.CSV"  # Bounding box annotations (not used in main)
    '''
    img_dir = "data/raw/images"  # Directory containing the images
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

    ## train_val_list = load_file_list(train_val_files)
    train_df, val_df, test_df = split_dataset(data_csv, val_size=0.15, test_size=0.15, random_state=42)

    # Create datasets, filtering by file lists
    train_dataset = ChestXrayDataset(train_df, img_dir, train_transforms, class_names)
    val_dataset = ChestXrayDataset(val_df, img_dir, val_transforms, class_names)
    test_dataset = ChestXrayDataset(test_df, img_dir, val_transforms, class_names)

    # Create DataLoaders for batching
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=2)  # Shuffles and batches training data
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, num_workers=2)  # Batches validation data without shuffling
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=2)  # Batches test data without shuffling

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # Get model, loss function, and optimizer
    # model = CustomCNN()
    model = CustomCNN(num_classes=len(class_names))
    model = model.to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    # Train the model
    train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device)  # Trains for num_epochs

    print("Training complete!")

    test_loss, test_acc = evaluate_model(model, test_loader, device, loss_fn)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "models/custom_cnn_chestxray.pth")
    print("Model saved to models/custom_cnn_chestxray.pth")

if __name__ == "__main__":
    main()