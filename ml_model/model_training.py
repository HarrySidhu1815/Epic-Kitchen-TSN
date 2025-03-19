import torch
import torch.nn as nn
import torch.optim as optim
from ml_model.tsn import TSN
from ml_model.dataset_preparartion import dataloader
from ml_model.dataset_preparartion import labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TSN(num_classes=len(set(labels))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for frames, labels in dataloader:
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")
