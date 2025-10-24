# Load pre-trained model and modify the final layer for regression.

from torchvision import models
import torch.nn as nn

def get_model(num_classes=14, freeze_features=True):
    
    # Load DenseNet-121 model pre-trained on ImageNet.
    model = models.densenet121(weights="IMAGENET1K_V1")

    if (freeze_features):
        # Freeze all pre-trained layers except the final classifier layer.
        for param in model.features.parameters():
            param.requires_grad = False

    # Modify the classifier to output `num_classes` classes.
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()  # For multi-label classification
    )

    return model