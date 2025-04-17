import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("attack_results", exist_ok=True)

# ----------------------------
# Define a Simple CNN
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # [batch, 32, 14, 14]
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----------------------------
# FGSM Attack Function
# ----------------------------
def fgsm_attack(image, epsilon, gradient):
    sign_data_grad = gradient.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# ----------------------------
# Training Function
# ----------------------------
def train_model(model, train_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} complete.")

# ----------------------------
# Test with FGSM Attack
# ----------------------------
def test_adversarial(model, test_loader, epsilon):
    correct = 0
    adv_examples = []

    model.eval()
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # Only attack correctly classified images
        correct_mask = init_pred.eq(target.view_as(init_pred))
        if not correct_mask.any():
            continue

        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        correct += final_pred.eq(target.view_as(final_pred)).sum().item()

        if len(adv_examples) < 5:
            adv_examples.append((data.squeeze().cpu(), perturbed_data.squeeze().cpu(), target.item(), final_pred.item()))

    final_acc = correct / len(test_loader.dataset)
    print(f"Epsilon: {epsilon}\tTest Accuracy = {final_acc:.4f}")
    return final_acc, adv_examples

# ----------------------------
# Visualization
# ----------------------------
def visualize_examples(examples, epsilon):
    fig, axes = plt.subplots(len(examples), 2, figsize=(6, 10))
    for i, (orig, adv, label, pred) in enumerate(examples):
        axes[i, 0].imshow(orig.detach().numpy(), cmap="gray")
        axes[i, 0].set_title(f"Original: {label}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(adv.detach().numpy(), cmap="gray")
        axes[i, 1].set_title(f"Adversarial: {pred}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"attack_results/fgsm_examples_epsilon_{epsilon:.2f}.png")
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    # Data
    transform = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1, shuffle=True
    )

    # Load or Train
    model = SimpleCNN().to(device)
    model_path = "cnn_mnist.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded trained model.")
    else:
        print("Training model...")
        train_model(model, train_loader)
        torch.save(model.state_dict(), model_path)

    # Attack with multiple epsilons
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]
    accuracies = []

    for eps in epsilons:
        acc, examples = test_adversarial(model, test_loader, eps)
        accuracies.append(acc)
        visualize_examples(examples, eps)

    # Plot accuracy vs epsilon
    plt.figure(figsize=(6, 4))
    plt.plot(epsilons, accuracies, marker='o')
    plt.title("Model Accuracy vs FGSM Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("attack_results/fgsm_accuracy_plot.png")
    plt.close()

if __name__ == "__main__":
    main()
