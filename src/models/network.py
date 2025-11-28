# Load pre-trained model and modify the final layer for regression.

from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
def get_model(num_classes=14, freeze_features=True):
    
    # Load DenseNet-121 model pre-trained on ImageNet.
    model = models.densenet121(weights="IMAGENET1K_V1")

    '''
    if (freeze_features):
        # Freeze all pre-trained layers except the final classifier layer.
        for param in model.features.parameters():
            param.requires_grad = False
    '''
    # Modify the classifier to output `num_classes` classes.
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model

class CustomCNN(nn.Module):
    def __init__(self, num_classes=14, dropout_rate=0.5, freeze_features=True):
        super().__init__()
        backbone = models.densenet121(weights="IMAGENET1K_V1")

        if freeze_features:
            for param in backbone.features.parameters():
                param.requires_grad = False
         
        self.features = backbone.features
        num_features = backbone.classifier.in_features
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
"""