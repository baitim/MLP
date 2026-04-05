import numpy as np
from mlp import MLP
from dataset import generate_dataset, split_dataset

def clip_gradients(grads, max_norm):
    if max_norm is None:
        return
    total_norm = 0.0
    for g in grads:
        if g is not None:
            total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for g in grads:
            if g is not None:
                g *= scale

def sgd_update(params, grads, lr, weight_decay):
    for p, g in zip(params, grads):
        if g is not None:
            np.nan_to_num(g, copy=False)
            p -= lr * (g + weight_decay * p)

def train_epoch(model, X, y, batch_size, lr, weight_decay, max_norm):
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    params, grads = model.parameters()
    total_loss = 0.0
    num_batches = 0
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        pred = model.forward(X_batch)
        loss = model.loss(pred, y_batch)
        total_loss += loss
        num_batches += 1
        model.backward()
        clip_gradients(grads, max_norm)
        sgd_update(params, grads, lr, weight_decay)
    return total_loss / num_batches

def evaluate(model, X, y):
    pred = model.forward(X)
    loss = model.loss(pred, y)
    return loss

def train(model, X_train, y_train, X_val, y_val,
          epochs, batch_size, lr, weight_decay, max_norm):
    for epoch in range(epochs):
        train_loss = train_epoch(model, X_train, y_train, batch_size, lr, weight_decay, max_norm)
        val_loss = evaluate(model, X_val, y_val)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 10000
    X, y = generate_dataset(n_samples)
    X_train, y_train, X_val, y_val = split_dataset(X, y, train_ratio=0.8)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_val = (X_val - X_mean) / (X_std + 1e-8)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    model = MLP(input_size=4, hidden_sizes=[128, 128], output_size=4)
    epochs = 250
    batch_size = 32
    lr = 0.001
    weight_decay = 1e-4
    max_norm = 5.0
    train(model, X_train, y_train_norm, X_val, y_val_norm,
          epochs, batch_size, lr, weight_decay, max_norm)
    y_pred_norm = model.forward(X_val)
    y_pred = y_pred_norm * y_std + y_mean
    mse_original = np.mean((y_pred - y_val) ** 2)
    print(f"\nFinal validation loss (original scale): {mse_original:.6f}")
    print("\nSample predictions (first 5 validation samples):")
    for i in range(5):
        print(f"True: {y_val[i]}, Pred: {y_pred[i]}")