import torch
from src.datasets.dataset import ChestXrayDataset
from src.datasets.dataloader import get_dataloader
from src.preprocessing.preprocess import get_transforms
from src.models.network import get_model
from src.training.loss import get_loss_function  ## TODO: Implement this
from src.training.train import train_model  ## TODO: Implement this

def main():
    train_csv = "data/train_labels.csv"  ## TODO: Update paths
    val_csv = "data/val_labels.csv"      ## TODO: Update paths
    img_dir = "data/images/"  ## TODO: Update paths
    num_epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU.
    lr = 1e-4  # Step size for optimizer

    # Get dataset and dataloaders.
    train_transforms = get_transforms(train=True)
    val_transforms = get_transforms(train=False)

    # Load datasets in readable format.
    class_names = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]
    train_dataset = ChestXrayDataset(train_csv, img_dir, train_transforms, class_names)
    val_dataset = ChestXrayDataset(val_csv, img_dir, val_transforms, class_names)

    # Create dataloaders for batching: select batch_size (32) samples for each iteration.
    train_loader = get_dataloader(train_dataset, batch_size=batch_size)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size)

    # Get model, loss function, and optimizer.
    model = get_model().to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model.
    train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device)

if __name__ == "__main__":
    main()