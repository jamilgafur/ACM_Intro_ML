import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

# Hyperparameters
batch_size = 128
z_dim = 100
lr = 0.0002
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("gan_images", exist_ok=True)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
])

dataloader = DataLoader(
    datasets.MNIST(root='./data', train=True, transform=transform, download=True),
    batch_size=batch_size, shuffle=True
)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()  # Output between [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Labels
real_label = 1.
fake_label = 0.

fixed_noise = torch.randn(64, z_dim, device=device)  # fixed seed to visualize progress

G_losses = []
D_losses = []

# Training Loop
print("Starting Training...")
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)

        # === Train Discriminator ===
        D.zero_grad()
        # Real images
        label = torch.full((real_imgs.size(0),), real_label, dtype=torch.float, device=device)
        output = D(real_imgs).squeeze()
        loss_D_real = criterion(output, label)
        loss_D_real.backward()

        # Fake images
        noise = torch.randn(real_imgs.size(0), z_dim, device=device)
        fake_imgs = G(noise)
        label.fill_(fake_label)
        output = D(fake_imgs.detach()).squeeze()
        loss_D_fake = criterion(output, label)
        loss_D_fake.backward()

        loss_D = loss_D_real + loss_D_fake
        optimizer_D.step()

        # === Train Generator ===
        G.zero_grad()
        label.fill_(real_label)  # Generator wants discriminator to believe it's real
        output = D(fake_imgs).squeeze()
        loss_G = criterion(output, label)
        loss_G.backward()
        optimizer_G.step()

    # Logging
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")
    G_losses.append(loss_G.item())
    D_losses.append(loss_D.item())

    # Save generated samples
    if (epoch + 1) % 10 == 0 or epoch == 0:
        with torch.no_grad():
            fake_imgs = G(fixed_noise).detach().cpu()
        vutils.save_image(fake_imgs, f"gan_images/fake_epoch_{epoch+1:03d}.png", normalize=True, nrow=8)

# Plot losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Epoch")
plt.ylabel("Loss")
