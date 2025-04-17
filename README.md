# Project README

This project explores various machine learning techniques, including polynomial interpolation, neural networks, convolutional neural networks (CNNs), adversarial machine learning, and generative adversarial networks (GANs). It is designed to provide a hands-on understanding of machine learning concepts through practical examples. The project is structured into several Python scripts, each tackling a specific topic, providing useful insights for those new to machine learning.

---

## Directory Structure

```
├── 1_interpolation.py
├── 2_neural_network.py
├── 3_cnn.py
├── 4_adversarial_ML.py
├── 5_MNIST_GAN.py
└── requirements.txt
```

### File Descriptions

---

### `1_interpolation.py`

This script demonstrates **polynomial interpolation**, a mathematical method used to approximate a function based on a set of data points. Polynomial interpolation can be useful when trying to fit a smooth curve to noisy or incomplete data.

#### Key Concepts:
- **Polynomial Interpolation**: This technique involves fitting a polynomial to a set of points. Given a set of points \((x_i, y_i)\), we seek a polynomial \(P(x)\) that minimizes the error between the predicted \(y\)-values and the actual observed \(y_i\)-values.
  
  Mathematically, the polynomial is expressed as:

  \[
  P(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + ... + \theta_d x^d
  \]

  where \(d\) is the degree of the polynomial, and \(\theta_i\) are the coefficients we aim to determine.

#### Code Explanation:
- **Data Generation**: The `generate_data` function creates synthetic data points using a quadratic function \(y = 0.5x^2 - x + 2\) and adds Gaussian noise.
- **Polynomial Feature Expansion**: The `polynomial_features` function generates polynomial features (powers of \(x\)) to transform the data for fitting the polynomial.
- **Polynomial Fit**: The `fit_polynomial` function uses **least squares estimation** to find the best-fitting polynomial coefficients \(\theta\).
- **Visualization**: The script then plots the noisy data and the fitted polynomial to visualize the interpolation.

This serves as an introduction to understanding how polynomial fitting works in the context of noisy data.

---

### `2_neural_network.py`

This file implements a basic **Neural Network** to approximate a quadratic function. It uses a multi-layer perceptron (MLP) architecture, a type of neural network composed of multiple layers of neurons.

#### Key Concepts:
- **Neural Networks (NNs)**: A neural network is a series of layers of neurons that transform the input data to a desired output. Each neuron computes a weighted sum of its inputs, applies an activation function, and passes the result to the next layer.
  
  A simple neural network might consist of:
  
  \[
  y = f(Wx + b)
  \]

  where \(W\) are the weights, \(b\) is the bias, \(x\) is the input, and \(f\) is the activation function.

- **Activation Function**: In this case, ReLU (Rectified Linear Unit) is used as the activation function, defined as:

  \[
  f(x) = \max(0, x)
  \]

  ReLU introduces non-linearity, allowing the network to learn complex patterns.

- **Training Process**: The network is trained using **backpropagation**, an algorithm for updating the weights by minimizing a loss function, typically Mean Squared Error (MSE) for regression problems.

#### Code Explanation:
- **Data Generation**: Similar to the previous script, synthetic noisy data is created using a quadratic function.
- **Model**: The `PolyNN` class defines a neural network with a configurable number of hidden layers and neurons. It uses ReLU activations between layers.
- **Training**: The model is trained using the **Adam optimizer** and **MSE loss**. The `train_model` function adjusts the weights to minimize the loss between predicted and actual values.
- **Evaluation**: The model's performance is evaluated using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and \(R^2\)-score, which gives an indication of the model's goodness-of-fit.
- **Visualization**: The script also plots the fitted curve and training metrics over epochs.

This file demonstrates how a neural network can be trained to learn complex relationships in data, even when the data is noisy.

---

### `3_cnn.py`

This script focuses on **Convolutional Neural Networks (CNNs)**, which are a class of deep neural networks most commonly used for image classification tasks.

#### Key Concepts:
- **CNN Layers**: CNNs use convolutional layers, pooling layers, and fully connected layers. The convolutional layers apply filters to extract features from images, such as edges, shapes, and textures. Pooling layers reduce the dimensionality, keeping only the most important features.
  
  The convolution operation in a CNN is defined as:

  \[
  y(x, y) = (f * g)(x, y) = \sum_m \sum_n f(m, n) g(x - m, y - n)
  \]

  where \(f\) is the input image and \(g\) is the filter (kernel).

- **Activation Function**: CNNs typically use ReLU activation to introduce non-linearity.
  
- **Pooling**: **Max-pooling** is used to down-sample the feature maps, reducing their dimensionality.

#### Code Explanation:
- **Data**: The script uses the **MNIST dataset**, a collection of handwritten digits, as the input data. The `transform` applies normalization and converts images to tensors.
- **Model**: The `MLP` class defines a simple CNN with two convolutional layers, followed by a few fully connected layers. The network is trained to classify MNIST digits into one of ten categories.
- **Training and Testing**: The model is trained over 10 epochs, and the loss and accuracy are tracked. The `train_model` function implements the training loop, while `test_adversarial` evaluates the model's accuracy.
- **Visualization**: Several functions visualize different aspects of the model's behavior, including plotting the loss and accuracy over epochs and displaying the model's predictions on test images.

This file demonstrates how CNNs excel at image classification tasks, especially when working with visual data like handwritten digits.

---

### `4_adversarial_ML.py`

This file explores the concept of **Adversarial Machine Learning**, which focuses on the vulnerability of machine learning models to small, intentional perturbations in the input data that can lead to incorrect predictions.

#### Key Concepts:
- **Adversarial Attacks**: The **Fast Gradient Sign Method (FGSM)** is one of the most common methods used for adversarial attacks. It generates perturbations by adding noise in the direction of the gradient of the loss function with respect to the input data.

  Mathematically:

  \[
  \delta = \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
  \]

  where \(J\) is the loss function, \(x\) is the input, \(y\) is the target label, \(\theta\) are the model parameters, and \(\epsilon\) is the perturbation magnitude.

#### Code Explanation:
- **Model**: A simple CNN is defined, similar to the one used in the previous script.
- **FGSM Attack**: The `fgsm_attack` function adds perturbations to the input image by taking the gradient of the loss with respect to the image and scaling it by a small factor \(\epsilon\).
- **Training**: The model is trained on MNIST images using the same process as in the previous script.
- **Adversarial Testing**: After training, the model is tested on adversarial examples generated using FGSM. The script evaluates the model's robustness by testing its accuracy on perturbed images.
- **Visualization**: The script also visualizes a few adversarial examples and compares the original and perturbed images.

This file demonstrates how adversarial attacks can compromise the performance of machine learning models, emphasizing the need for robust models in real-world applications.

---

### `5_MNIST_GAN.py`

This script introduces **Generative Adversarial Networks (GANs)**, a class of machine learning models used to generate new data instances that resemble a given training dataset. GANs consist of two networks:
1. **Generator**: This network creates fake data (images, for example).
2. **Discriminator**: This network tries to differentiate between real and fake data.

The generator and discriminator are trained simultaneously, with the generator learning to create better data to fool the discriminator.

#### Key Concepts:
- **GANs**: GANs work by using a **min-max game** between the generator and discriminator. The generator tries to minimize the difference between generated data and real data, while the discriminator tries to maximize its ability to distinguish between real and fake data.

  The objective for GANs is:

  \[
  \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{\text{z}}(z)}[\log(1 - D(G(z)))]
  \]

  where \(D\) is the discriminator, \(G\) is the generator, and \(z\) is the noise input to the generator.

#### Code Explanation:
This file implements a GAN that generates MNIST-like digits using a simple neural network architecture for both the generator and discriminator. It trains the GAN on the MNIST dataset, allowing the generator to learn how to create realistic images of handwritten digits.
