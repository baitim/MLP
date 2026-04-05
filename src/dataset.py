import numpy as np

def target_function(x):
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    y1 = x1 ** 2
    y2 = 3 * x2
    y3 = 5 * x4 - x3
    y4 = np.full_like(x1, 3)
    return np.stack([y1, y2, y3, y4], axis=1)

def generate_dataset(n_samples, input_mean=0, input_std=1):
    X = np.random.normal(input_mean, input_std, size=(n_samples, 4))
    y = target_function(X)
    return X, y

def split_dataset(X, y, train_ratio=0.8):
    n = len(X)
    indices = np.random.permutation(n)
    train_n = int(n * train_ratio)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]