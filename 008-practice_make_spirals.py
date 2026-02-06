import math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

# Spirals create deep computation graphs - increase recursion limit
sys.setrecursionlimit(10000)


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only supporting int/float powers"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')
        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, num_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, num_inputs, layer_sizes):
        sizes = [num_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def generate_spirals(n_per_class=100, noise=0.3, seed=42):
    """Generate two interleaving spirals."""
    np.random.seed(seed)
    theta = np.linspace(0, 4 * np.pi, n_per_class)

    # Spiral 1
    r1 = theta + np.random.randn(n_per_class) * noise
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)

    # Spiral 2 (rotated by pi)
    r2 = theta + np.random.randn(n_per_class) * noise
    x2 = r2 * np.cos(theta + np.pi)
    y2 = r2 * np.sin(theta + np.pi)

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])

    # Normalize - critical for tanh gradients
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y


def compute_loss(model, X, y, alpha=5e-5):
    """Compute hinge loss with L2 regularization."""
    inputs = [list(row) for row in X]
    predictions = [model(x) for x in inputs]

    losses = []
    for yi, pred in zip(y, predictions):
        margin = 1 + (-yi) * pred
        if margin.data > 0:
            losses.append(margin)
        else:
            losses.append(Value(0))

    avg_data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = alpha * sum(p*p for p in model.parameters())
    total_loss = avg_data_loss + reg_loss

    accuracy = sum(1 for yi, pred in zip(y, predictions) if (yi > 0) == (pred.data > 0)) / len(y)
    return total_loss, accuracy


def plot_decision_boundary(model, X, y, accuracy):
    """Plot the decision boundary learned by the model."""
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = [model(list(point)).data for point in mesh_points]
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=np.linspace(-1, 1, 20), cmap='RdBu', alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Neural Network Decision Boundary - Spirals (Accuracy: {accuracy*100:.1f}%)')
    plt.colorbar(label='Prediction')
    plt.show()


def train(model, X, y, num_iterations=1000, patience=100, target_accuracy=0.99):
    """Train with warmup, cosine decay, gradient clipping, and early stopping."""
    print("Training...")
    print("=" * 60)

    best_accuracy = 0
    patience_counter = 0

    for iteration in range(num_iterations):
        # Learning rate: warmup then cosine decay
        if iteration < 50:
            lr = 0.01 + (1.0 - 0.01) * (iteration / 50)
        else:
            t = (iteration - 50) / (num_iterations - 50)
            lr = 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * t))

        loss, accuracy = compute_loss(model, X, y)

        # Early stopping check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at iteration {iteration}")
            print(f"Best accuracy: {best_accuracy*100:.2f}%")
            break

        # Zero gradients
        for p in model.parameters():
            p.grad = 0.0

        loss.backward()

        # Gradient clipping
        max_grad = 5.0
        for p in model.parameters():
            p.grad = max(-max_grad, min(max_grad, p.grad))

        # Update weights
        for p in model.parameters():
            p.data -= lr * p.grad

        if iteration % 20 == 0:
            print(f"Iter {iteration:4d} | Loss: {loss.data:.4f} | Acc: {accuracy*100:5.1f}% | LR: {lr:.4f} | Patience: {patience_counter}/{patience}")

        if accuracy >= target_accuracy:
            print(f"\nTarget accuracy reached at iteration {iteration}!")
            break

    print("=" * 60)
    print(f"Final - Loss: {loss.data:.4f} | Accuracy: {accuracy*100:.1f}%")
    print(f"Best  - Accuracy: {best_accuracy*100:.1f}%")
    return loss, accuracy


if __name__ == "__main__":
    # Generate spirals dataset
    X, y = generate_spirals(n_per_class=100, noise=0.3, seed=42)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Labels: {np.unique(y)}")

    # Create model - larger network for complex spiral boundary
    random.seed(42)
    model = MLP(2, [32, 32, 1])
    print(f"Model parameters: {len(model.parameters())}")

    # Train with advanced techniques
    loss, accuracy = train(model, X, y)

    # Visualize
    plot_decision_boundary(model, X, y, accuracy)
