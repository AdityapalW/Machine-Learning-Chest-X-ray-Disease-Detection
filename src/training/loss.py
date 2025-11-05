# Implement custom loss function.
import torch.nn as nn

def get_loss_function():
    """
    Returns the loss function for binary classification.
    In this case, we use BCEWithLogitsLoss, which is suitable for binary classification tasks.
    """
    return nn.BCEWithLogitsLoss()