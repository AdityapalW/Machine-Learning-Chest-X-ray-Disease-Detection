import torch

def evaluate_model(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device).float()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / (total_samples * labels.size(1))

    return avg_loss, accuracy