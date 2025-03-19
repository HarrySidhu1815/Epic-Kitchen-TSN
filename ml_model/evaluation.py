from ml_model.model_training import model
from ml_model.dataset_preparartion import dataloader
from ml_model.model_training import device
import torch

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for frames, labels in dataloader:
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
