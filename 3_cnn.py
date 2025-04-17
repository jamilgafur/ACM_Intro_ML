import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Set up
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("figures", exist_ok=True)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),  # Normalize [0, 255] to [0.0, 1.0]
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Visualize a batch
def plot_sample_batch():
    images, labels = next(iter(train_loader))
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(images[i][0], cmap='gray')
        axes[i].set_title(f"{labels[i].item()}")
        axes[i].axis('off')
    plt.suptitle("Sample MNIST Digits")
    plt.tight_layout()
    plt.savefig("figures/mnist_batch.png")
    plt.close()

plot_sample_batch()

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Metrics
train_loss_hist, test_loss_hist = [], []
train_acc_hist, test_acc_hist = [], []

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

# Training Loop
for epoch in range(10):
    model.train()
    running_loss, running_acc = 0.0, 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = compute_accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc

    avg_train_loss = running_loss / len(train_loader)
    avg_train_acc = running_acc / len(train_loader)
    train_loss_hist.append(avg_train_loss)
    train_acc_hist.append(avg_train_acc)

    # Validation
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = compute_accuracy(outputs, labels)

            test_loss += loss.item()
            test_acc += acc

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    test_loss_hist.append(avg_test_loss)
    test_acc_hist.append(avg_test_acc)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
          f"Test Loss: {avg_test_loss:.4f}, Acc: {avg_test_acc:.4f}")

# Save Loss & Accuracy plots
def plot_metrics(train_loss, test_loss, train_acc, test_acc):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, test_acc, label="Test Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("figures/metrics.png")
    plt.close()

plot_metrics(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist)

# Save predictions from test set
def plot_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i in range(8):
        axes[0, i].imshow(images[i][0].cpu(), cmap="gray")
        axes[0, i].set_title(f"True: {labels[i].item()}")
        axes[0, i].axis("off")

        axes[1, i].imshow(images[i][0].cpu(), cmap="gray")
        axes[1, i].set_title(f"Pred: {preds[i].item()}")
        axes[1, i].axis("off")

    plt.suptitle("Model Predictions on MNIST Test Set")
    plt.tight_layout()
    plt.savefig("figures/predictions.png")
    plt.close()

plot_predictions()
