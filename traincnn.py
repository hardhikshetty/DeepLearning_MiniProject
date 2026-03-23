import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from simplecnn import SimpleCNN
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

print("Code started...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Data Augmentation + Normalization -----
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # flip for augmentation
    transforms.RandomCrop(32, padding=4), # crop with padding
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

print("Loading dataset...")

train_data = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

test_data = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)  # bigger batch for faster training
test_loader = DataLoader(test_data, batch_size=128)

# ----- Model, Loss, Optimizer -----
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []

# ----- Training Loop -----
num_epochs = 5  # reduced epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
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
correct = 0
total = 0
all_preds = []
all_labels = []

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
print("Confusion Matrix:\n", cm)

# ----- Loss Plot -----
plt.figure(figsize=(6,4))
plt.plot(train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.savefig("cnn_loss.png")  # save in local folder
plt.show()