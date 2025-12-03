import torch
import matplotlib.pyplot as plt

checkpoint = torch.load("checkpoint.pth", map_location="cpu")

train_losses = checkpoint["train_loss"]
val_losses = checkpoint["val_loss"]

plt.figure(figsize=(6,4))
epochs = range(1, len(train_losses)+1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

train_acc=checkpoint["train_acc"]
val_acc=checkpoint["val_acc"]

plt.figure(figsize=(6,4))
epochs = range(1, len(train_acc)+1)
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
