# Resize images to 224*224
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize from 1024*1024 to 224*224 for scalability
    transforms.RandomHorizontalFlip(),  # Data augmentation to prevent overfitting
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # Normalize as per pre-trained model requirements
])
