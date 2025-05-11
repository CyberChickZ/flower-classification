import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='data/val', transform=transform)

# Data loader
train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=15)

print(f"Number of classes in the training set: {len(train_dataset.classes)}")
print(f"Class names: {train_dataset.classes}")

# Load pre-trained ResNet18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Replace output layer: from 1000 classes to 5 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# print(model)

# Loss function: CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam is more stable
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5  # Train for 5 epochs

num_epochs = 5

for epoch in range(num_epochs):
    model.train()  # Set model to training mode (affects Dropout, BatchNorm)
    running_loss = 0.0  # Accumulate total training loss
    correct_train = 0   # Count correctly predicted samples in training

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to same device as model

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass + parameter update (standard training steps)
        optimizer.zero_grad()   # Clear previous gradients
        loss.backward()         # Compute gradients from loss
        optimizer.step()        # Update model parameters

        # Accumulate training loss and correct predictions
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)  # Take highest score index as predicted class
        correct_train += (preds == labels).sum().item()

        # Optional: print training progress every few batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  [Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # Compute average training loss and accuracy
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_train / len(train_loader.dataset)

    # ========== Validation loop ==========
    model.eval()  # Switch to eval mode (disable Dropout, use running stats in BN)
    correct_val = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()

    val_acc = correct_val / len(val_loader.dataset)

    # Print training progress after each epoch
    print(f"📘 Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}  Val Acc: {val_acc:.4f}")

print("Training completed!")

# Save model
torch.save(model.state_dict(), "flower_resnet18.pth")
print("Model saved to flower_resnet18.pth")
