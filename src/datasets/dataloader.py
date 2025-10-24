# Define DataLoader for batching and shuffling the dataset.

from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size=32, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)