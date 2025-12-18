# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:10:39 2025

@author: 18576
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Redirect console output to file
log_file = open("experiment_log.txt", "w")
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = log_file
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
sys.stdout = Logger()

device = torch.device("cpu")

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32*32*3, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SoftmaxClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 10
train_losses, test_accuracies = [], []
classes = train_dataset.classes

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Test accuracy
    model.eval()
    correct, total = 0, 0
    misclassified = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Collect misclassified samples
            for img, pred, actual in zip(images, predicted, labels):
                if pred != actual and len(misclassified) < 8:
                    misclassified.append((img.cpu(), pred.item(), actual.item()))
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# Visualization:
plt.subplot(1,2,1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.subplot(1,2,2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig("training_results.png")
plt.show()

# Probability Visualization for one sample
sample_img, sample_label = test_dataset[0]
model.eval()
with torch.no_grad():
    output = model(sample_img.unsqueeze(0))
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
plt.figure(figsize=(8,4))
plt.bar(classes, probs)
plt.xticks(rotation=45)
plt.title(f"Probability Distribution (True: {classes[sample_label]})")
plt.savefig("probability_visualization.png")
plt.show()

# Misclassified Samples Visualization
fig, axes = plt.subplots(2, 4, figsize=(12,6))
for i, (img, pred, actual) in enumerate(misclassified):
    ax = axes[i//4, i%4]
    img = img / 2 + 0.5
    npimg = img.numpy().transpose((1,2,0))
    ax.imshow(npimg)
    ax.set_title(f"P:{classes[pred]} | A:{classes[actual]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("misclassified_samples.png")
plt.show()

log_file.close()