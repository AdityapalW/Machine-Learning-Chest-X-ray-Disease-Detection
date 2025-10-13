# Load pre-trained model and modify the final layer for regression.

from torchvision import models
import torch.nn as nn

# Load DenseNet-121 model pre-trained on ImageNet.
model = models.densenet121(weights="IMAGENET1K_V1")

# Freeze all pre-trained layers except the final classifier layer.
for param in model.features.parameters():
    param.requires_grad = False

# Modify the classifier to output 14 classes instead of 1000 from ImageNet.
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 14)