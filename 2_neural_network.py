import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

def generate_data(num_points=30, noise_std=0.3):
    x = torch.linspace(-3, 3, num_points).unsqueeze(1)
    y_true = 0.5 * x**2 - x + 2
    y_noisy = y_true + noise_std * torch.randn_like(y_true)
    return x, y_noisy

class PolyNN(nn.Module):
    def __init__(self, hidden_dim=64, depth=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def compute_metrics(y_pred, y_true):
    mse = torch.mean((y_pred - y_true)**2).item()
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    ss_res = torch.sum((y_true - y_pred)**2)
    r2 = 1 - ss_res / ss_tot
    return {'MSE': mse, 'MAE': mae, 'R2': r2.item()}

def train_model(model, x, y, epochs=1000, lr=1e-2):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {'loss': [], 'r2': [], 'mae': []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics = compute_metrics(y_pred, y)
            history['loss'].append(loss.item())
            history['r2'].append(metrics['R2'])
            history['mae'].append(metrics['MAE'])

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | R²: {metrics['R2']:.4f} | MAE: {metrics['MAE']:.4f}")

    return history

def plot_fit(x, y, model):
    x_fit = torch.linspace(x.min(), x.max(), 200).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        y_fit = model(x_fit)

    plt.figure(figsize=(8, 5))
    plt.scatter(x.numpy(), y.numpy(), label='Noisy Data', color='blue')
    plt.plot(x_fit.numpy(), y_fit.numpy(), label='NN Fit', color='red')
    plt.title('Neural Network Interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('nn_interpolation.png')

def plot_training_metrics(history):
    epochs = range(len(history['loss']))
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Loss (MSE)')
    plt.plot(epochs, history['mae'], label='MAE')
    plt.xlabel('Epoch')
    plt.title('Loss & MAE over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['r2'], label='R² Score', color='green')
    plt.xlabel('Epoch')
    plt.title('R² Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')

# ---- Main ----
if __name__ == "__main__":
    x, y = generate_data(num_points=30, noise_std=0.3)
    model = PolyNN(hidden_dim=64, depth=3)
    history = train_model(model, x, y, epochs=1000, lr=0.01)
    plot_fit(x, y, model)
    plot_training_metrics(history)
