import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

def generate_data(num_points=20, noise_std=0.1):
    # Generate evenly spaced x-values
    x = torch.linspace(-3, 3, num_points)
    # Create some underlying ground truth (e.g., quadratic)
    y_true = 0.5 * x**2 - x + 2
    # Add Gaussian noise
    y_noisy = y_true + noise_std * torch.randn_like(y_true)
    return x, y_noisy

def polynomial_features(x, degree):
    """
    Given input tensor x, return polynomial features up to given degree.
    Example: x=[1,2], degree=2 -> [[1,1], [1,2], [1,4]]
    """
    return torch.stack([x**i for i in range(degree + 1)], dim=1)

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of given degree to (x, y) using least squares in PyTorch.
    """
    X = polynomial_features(x, degree)      # Design matrix: shape [N, degree+1]
    y = y.unsqueeze(1)                      # Make y shape [N, 1]

    result = torch.linalg.lstsq(X, y)       # X @ theta ≈ y
    theta = result.solution.squeeze()       # Get solution and squeeze to 1D
    return theta


def predict(x, theta):
    """
    Predict using fitted polynomial coefficients theta.
    """
    X = polynomial_features(x, len(theta) - 1)
    return X @ theta

def plot_fit(x, y, theta, degree):
    x_fit = torch.linspace(x.min(), x.max(), 200)
    y_fit = predict(x_fit, theta)

    plt.figure(figsize=(8, 5))
    plt.scatter(x.numpy(), y.numpy(), label='Noisy Data', color='blue')
    plt.plot(x_fit.numpy(), y_fit.detach().numpy(), label=f'Degree-{degree} Fit', color='red')
    plt.title(f'Polynomial Interpolation (degree={degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('polynomial_interpolation.png')

# ----- Run Example -----
if __name__ == "__main__":
    x, y = generate_data(num_points=20, noise_std=0.5)

    degree = 10  # Change this to 1 for linear, 2 for quadratic, etc.
    theta = fit_polynomial(x, y, degree)
    plot_fit(x, y, theta, degree)
