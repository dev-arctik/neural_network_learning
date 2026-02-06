import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


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


def compute_loss(model, X, y, alpha=1e-4):
    """Compute hinge loss with L2 regularization."""
    inputs = [list(row) for row in X]
    predictions = [model(x) for x in inputs]

    # Hinge loss: max(0, 1 - y * prediction)
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


def plot_decision_boundary(model, X, y):
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
    plt.title('Neural Network Decision Boundary - Circles')
    plt.colorbar(label='Prediction')
    plt.show()


def train(model, X, y, learning_rate=1.0, num_iterations=100, target_accuracy=0.99):
    """Train the model using gradient descent."""
    print("Training...")
    print("=" * 50)

    for iteration in range(num_iterations):
        loss, accuracy = compute_loss(model, X, y)

        for p in model.parameters():
            p.grad = 0.0

        loss.backward()

        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if iteration % 10 == 0:
            print(f"Iteration {iteration:3d} | Loss: {loss.data:.4f} | Accuracy: {accuracy*100:.1f}%")

        if accuracy >= target_accuracy:
            print(f"Target accuracy reached! | Loss: {loss.data:.4f} | Accuracy: {accuracy*100:.1f}%")
            break

    print("=" * 50)
    print(f"Final - Loss: {loss.data:.4f} | Accuracy: {accuracy*100:.1f}%")
    return loss, accuracy


if __name__ == "__main__":
    # Generate dataset - concentric circles
    X, y = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
    y = y * 2 - 1  # Convert {0, 1} to {-1, +1}

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels: {np.unique(y)}")

    # Create model
    random.seed(17)
    model = MLP(2, [16, 16, 1])
    print(f"Model parameters: {len(model.parameters())}")

    # Train
    train(model, X, y)

    # Visualize
    plot_decision_boundary(model, X, y)
