import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from ganmodels import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs/gan_images", exist_ok=True)

#Hyperparameters
z_dim = 100
lr = 0.0002
batch_size = 64
num_epochs = 20  # reduced
img_dim = 28*28

#dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

G = Generator(z_dim=z_dim, img_dim=img_dim).to(device)
D = Discriminator(img_dim=img_dim).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

#Training 
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.view(-1, img_dim).to(device)
        batch_size_curr = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size_curr, 1).to(device)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)

        # Train Discriminator
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size_curr, z_dim).to(device)
        fake_images = G(z)
        outputs_fake = D(fake_images.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        #  Train Generator 
        z = torch.randn(batch_size_curr, z_dim).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        fake_images = G(z).view(-1, 1, 28, 28)
        plt.figure(figsize=(4,4))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(fake_images[i].cpu().squeeze()*0.5 + 0.5, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"outputs/gan_images/epoch_{epoch+1}.png")
        plt.close()