import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation
import os
from functools import partial

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        # Set random seed for reproducibility
        np.random.seed(0)
        
        # Save parameters
        self.lr = lr
        self.activation_fn = activation
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
        # Store activations and gradients
        self.z1 = None
        self.h = None  # This will be our hidden_output
        self.z2 = None
        self.output = None
        self.gradients = {'dW1': None, 'dW2': None}
    
    def _activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            x = np.clip(x, -100, 100)
            return 1 / (1 + np.exp(-x))
        raise ValueError(f"Unsupported activation: {self.activation_fn}")
    
    def _activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        raise ValueError(f"Unsupported activation: {self.activation_fn}")

    def forward(self, X):
        """Perform forward pass and store activations."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.h = self._activation(self.z1)  # This is our hidden_output
        self.z2 = np.dot(self.h, self.W2) + self.b2
        self.output = np.tanh(self.z2)  # Using tanh for final activation
        return self.output

    def backward(self, X, y):
        """Perform backward pass and store gradients."""
        batch_size = X.shape[0]
        
        # Compute output layer gradients
        d_output = self.output - y
        d_W2 = np.dot(self.h.T, d_output) / batch_size
        d_b2 = np.sum(d_output, axis=0, keepdims=True) / batch_size
        
        # Compute hidden layer gradients
        d_hidden = np.dot(d_output, self.W2.T) * self._activation_derivative(self.z1)
        d_W1 = np.dot(X.T, d_hidden) / batch_size
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True) / batch_size
        
        # Store gradients for visualization
        self.gradients['dW1'] = d_W1
        self.gradients['dW2'] = d_W2
        
        # Update weights and biases
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

def generate_data(n_samples=100):
    """Generate circular boundary dataset."""
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int).reshape(-1, 1)
    y = y * 2 - 1  # Convert to -1, 1 labels
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, total_steps):
    """Update function for animation."""
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    
    # Perform forward and backward passes
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Hidden space visualization
    hidden_features = mlp.h
    if hidden_features.shape[1] >= 3:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], 
                         c=y.ravel(), cmap='bwr', alpha=0.7)
        ax_hidden.set_title(f"Hidden Space at Step {frame}")
        ax_hidden.set_xlim(-1, 1)
        ax_hidden.set_ylim(-1, 1)
        ax_hidden.set_zlim(-1, 1)
    else:
        ax_hidden.text(0.5, 0.5, "Hidden layer dim < 3", fontsize=12, ha='center')

    # Input space visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)

    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], cmap='bwr', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame * 10}")

    # Gradient visualization with labels
    dW1 = mlp.gradients['dW1']
    dW2 = mlp.gradients['dW2']
    
    # Normalize gradients for visualization
    max_grad = max(np.abs(dW1).max(), np.abs(dW2).max())
    lw_scale = 5 / (max_grad + 1e-10)
    
    # Input layer nodes (normalized positions)
    input_y = np.linspace(0, 1, dW1.shape[0])
    # Hidden layer nodes (normalized positions)
    hidden_y = np.linspace(0, 1, dW1.shape[1])
    # Output layer node (single output)
    output_y = 0.5
    
    # Draw connections from input to hidden layer
    for i, iy in enumerate(input_y):
        for j, hy in enumerate(hidden_y):
            ax_gradient.plot([0, 0.5], [iy, hy], 'k-', 
                           alpha=0.5, 
                           lw=np.abs(dW1[i, j]) * lw_scale)
    
    # Draw connections from hidden to output layer
    for i, hy in enumerate(hidden_y):
        ax_gradient.plot([0.5, 1], [hy, output_y], 'k-', 
                        alpha=0.5, 
                        lw=np.abs(dW2[i, 0]) * lw_scale)
    
    # Plot nodes with labels
    # Input nodes
    for i, y_pos in enumerate(input_y):
        ax_gradient.scatter([0], [y_pos], s=100, c='blue', zorder=5)
        ax_gradient.annotate(f'x{i+1}', (0, y_pos), xytext=(-0.1, y_pos),
                           ha='right', va='center')
    
    # Hidden nodes
    for i, y_pos in enumerate(hidden_y):
        ax_gradient.scatter([0.5], [y_pos], s=100, c='green', zorder=5)
        ax_gradient.annotate(f'h{i+1}', (0.5, y_pos), xytext=(0.4, y_pos),
                           ha='right', va='center')
    
    # Output node
    ax_gradient.scatter([1], [output_y], s=100, c='red', zorder=5)
    ax_gradient.annotate('y', (1, output_y), xytext=(1.1, output_y),
                        ha='left', va='center')
    
    ax_gradient.set_title("Network Architecture and Gradients")
    ax_gradient.set_xticks([0, 0.5, 1])
    ax_gradient.set_xticklabels(['Input', 'Hidden', 'Output'])
    ax_gradient.set_yticks([])
    ax_gradient.grid(False)
    # Set proper limits to show labels
    ax_gradient.set_xlim(-0.2, 1.2)

def visualize(activation, lr, step_num):
    """Main visualization function."""
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, 
                                   ax_hidden=ax_hidden, ax_gradient=ax_gradient, 
                                   X=X, y=y, total_steps=step_num), 
                       frames=step_num//10, repeat=False)

    # Save the animation
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    ani.save(os.path.join(result_dir, "visualizeDemo.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)