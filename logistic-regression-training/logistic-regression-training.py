import numpy as np

def train_logistic_regression(X, y, lr, steps):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N, D = X.shape

    # Initialize parameters
    w = np.zeros(D)
    b = 0.0

    def _sigmoid(z):
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    for _ in range(steps):

        # Forward pass
        z = X @ w + b
        p = _sigmoid(z)

        # Gradients
        error = p - y

        dw = (X.T @ error) / N
        db = np.mean(error)

        # Parameter update
        w -= lr * dw
        b -= lr * db

    return (w, float(b))