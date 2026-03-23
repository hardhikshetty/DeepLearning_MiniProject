import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18, ResNet18_Weights

print("Starting Transfer Learning...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs/plots", exist_ok=True)

# ----- Data -----
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

# Use smaller subset for faster experiments
train_data = torch.utils.data.Subset(train_data, range(20000))
test_data = torch.utils.data.Subset(test_data, range(1000))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # bigger batch
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ----- Model -----
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze all layers except last layer and optionally last block
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block (layer4) for slightly better learning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final fully connected layer
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(model.fc.parameters()) + list(model.layer4.parameters()), 
    lr=0.0005
)

# ----- Training -----
train_losses = []
num_epochs = 5  # reduced from 15

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# ----- Evaluation -----
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# ----- Confusion Matrix -----
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=train_data.dataset.classes, 
            yticklabels=train_data.dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CIFAR-10 Confusion Matrix")
plt.savefig("outputs/plots/resnet_confusion_matrix.png")
plt.show()

# ----- Loss Plot -----
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss')
plt.title("ResNet18 Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("outputs/plots/resnet_loss.png")
plt.show()