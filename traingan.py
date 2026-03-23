print("Starting GAN training on Fashion-MNIST...")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

from gan.gan_model import Generator, Discriminator  # your GAN models

# ----- Folders -----
os.makedirs("outputs/generated_images", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ----- Dataset -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# ----- Models -----
z_dim = 100
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

# ----- Loss + Optimizers -----
criterion = nn.BCELoss()
lr = 0.0002
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

# ----- Training -----
num_epochs = 20  # reduced from 50
g_losses, d_losses = [], []

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, -1).to(device)

        # Real / Fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ----- Train Discriminator -----
        outputs_real = D(imgs)
        d_loss_real = criterion(outputs_real, real_labels)

        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z)
        outputs_fake = D(fake_imgs.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ----- Train Generator -----
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z)
        outputs = D(fake_imgs)
        g_loss = criterion(outputs, real_labels)  # trick: label fake as real
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    # Save losses
    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save generated images every 5 epochs
    if (epoch+1) % 5 == 0:
        fake_imgs_ = fake_imgs.view(-1, 1, 28, 28)
        grid = torchvision.utils.make_grid(fake_imgs_, nrow=8, normalize=True)
        plt.figure(figsize=(6,6))
        plt.imshow(grid.permute(1,2,0).cpu().detach())
        plt.title(f"Epoch {epoch+1}")
        plt.axis('off')
        plt.savefig(f"outputs/generated_images/epoch_{epoch+1}.png")
        plt.close()

# Plot final G/D losses
plt.figure()
plt.plot(range(1, num_epochs+1), g_losses, label='Generator Loss')
plt.plot(range(1, num_epochs+1), d_losses, label='Discriminator Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.grid(True)
plt.savefig("outputs/plots/gan_losses.png")
plt.show()