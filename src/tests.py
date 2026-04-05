import numpy as np
from mlp import Linear, ReLU, MSELoss, Sequential, MLP
from dataset import target_function

def test_linear_forward_backward():
    np.random.seed(0)
    linear = Linear(3, 2)
    x = np.random.randn(4, 3)
    out = linear.forward(x)
    expected_out = x @ linear.W + linear.b
    assert np.allclose(out, expected_out)
    grad_out = np.random.randn(4, 2)
    grad_in = linear.backward(grad_out)
    expected_grad_W = x.T @ grad_out
    expected_grad_b = np.sum(grad_out, axis=0)
    expected_grad_in = grad_out @ linear.W.T
    print("test_linear_forward_backward passed")

def test_relu_forward_backward():
    relu = ReLU()
    x = np.array([[-1, 2, 0], [3, -4, 5]])
    out = relu.forward(x)
    expected_out = np.maximum(x, 0)
    assert np.allclose(out, expected_out)
    grad_out = np.random.randn(2, 3)
    grad_in = relu.backward(grad_out)
    expected_grad_in = grad_out * (x > 0)
    assert np.allclose(grad_in, expected_grad_in)
    print("test_relu_forward_backward passed")

def test_mse_loss():
    mse = MSELoss()
    pred = np.array([[1, 2], [3, 4]])
    target = np.array([[1, 3], [5, 6]])
    loss = mse.forward(pred, target)
    expected_loss = np.mean((pred - target) ** 2)
    assert np.allclose(loss, expected_loss)
    grad = mse.backward()
    expected_grad = 2 * (pred - target) / pred.shape[0]
    assert np.allclose(grad, expected_grad)
    print("test_mse_loss passed")

def test_mlp_forward_shape():
    model = MLP(4, [8, 6], 4)
    x = np.random.randn(10, 4)
    out = model.forward(x)
    assert out.shape == (10, 4)
    print("test_mlp_forward_shape passed")

def test_training_convergence():
    np.random.seed(42)
    from dataset import generate_dataset, split_dataset
    X, y = generate_dataset(1000)
    X_train, y_train, X_val, y_val = split_dataset(X, y, train_ratio=0.7)
    model = MLP(4, [16, 16], 4)
    from train import train_epoch, evaluate
    params, grads = model.parameters()
    for g in grads:
        if g is not None:
            g.fill(0)
    initial_val_loss = evaluate(model, X_val, y_val)
    
    for epoch in range(30):
        train_loss = train_epoch(model, X_train, y_train, batch_size=16, lr=0.01, weight_decay=1e-5, max_norm=1.0)
    
    final_val_loss = evaluate(model, X_val, y_val)
    
    assert final_val_loss < initial_val_loss, f"Loss did not decrease: initial {initial_val_loss:.6f}, final {final_val_loss:.6f}"
    
    print(f"test_training_convergence passed (initial loss: {initial_val_loss:.6f} → final: {final_val_loss:.6f})")

def test_regularization_update():
    from train import sgd_update
    params = [np.array([1.0, 2.0]), np.array([3.0])]
    grads = [np.array([0.1, 0.2]), np.array([0.3])]
    lr = 0.1
    weight_decay = 0.01
    sgd_update(params, grads, lr, weight_decay)
    expected_p1 = np.array([1.0, 2.0]) - 0.1 * (np.array([0.1, 0.2]) + 0.01 * np.array([1.0, 2.0]))
    expected_p2 = np.array([3.0]) - 0.1 * (np.array([0.3]) + 0.01 * np.array([3.0]))
    assert np.allclose(params[0], expected_p1)
    assert np.allclose(params[1], expected_p2)
    print("test_regularization_update passed")

def test_gradient_accumulation():
    model = MLP(4, [4], 4)
    X = np.random.randn(8, 4)
    y = target_function(X)
    params, grads = model.parameters()
    for g in grads:
        if g is not None:
            g.fill(0)
    pred = model.forward(X[:4])
    loss1 = model.loss(pred, y[:4])
    model.backward()
    grads1 = [g.copy() if g is not None else None for g in grads]
    for g in grads:
        if g is not None:
            g.fill(0)
    pred2 = model.forward(X[4:])
    loss2 = model.loss(pred2, y[4:])
    model.backward()
    grads2 = [g.copy() if g is not None else None for g in grads]
    for g in grads:
        if g is not None:
            g.fill(0)
    for g1, g2 in zip(grads1, grads2):
        if g1 is not None:
            g1 += g2
    pred_full = model.forward(X)
    loss_full = model.loss(pred_full, y)
    model.backward()
    grads_full = [g.copy() if g is not None else None for g in grads]
    for g1, gf in zip(grads1, grads_full):
        if g1 is not None:
            assert np.allclose(g1, gf)
    print("test_gradient_accumulation passed")

if __name__ == "__main__":
    test_linear_forward_backward()
    test_relu_forward_backward()
    test_mse_loss()
    test_mlp_forward_shape()
    test_regularization_update()
    test_gradient_accumulation()
    test_training_convergence()
    print("All tests passed.")