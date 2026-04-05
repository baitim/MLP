import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        batch_size = self.x.shape[0]
        self.grad_W = (self.x.T @ grad_out) / batch_size
        self.grad_b = np.sum(grad_out, axis=0) / batch_size
        grad_in = grad_out @ self.W.T
        return grad_in

    def parameters(self):
        return [self.W, self.b], [self.grad_W, self.grad_b]


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask

    def parameters(self):
        return [], []


class MSELoss:
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        batch_size = self.pred.shape[0]
        return 2 * (self.pred - self.target) / batch_size


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        grads = []
        for layer in self.layers:
            p, g = layer.parameters()
            params.extend(p)
            grads.extend(g)
        return params, grads


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(Linear(prev, h))
            layers.append(ReLU())
            prev = h
        layers.append(Linear(prev, output_size))
        self.net = Sequential(layers)
        self.loss_fn = MSELoss()

    def forward(self, x):
        return self.net.forward(x)

    def loss(self, pred, target):
        return self.loss_fn.forward(pred, target)

    def backward(self):
        grad = self.loss_fn.backward()
        self.net.backward(grad)

    def parameters(self):
        return self.net.parameters()