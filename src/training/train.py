# Main training loop


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device):
    '''
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_value = loss_fn(outputs, labels)
            loss_value.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value.item()}")
    '''