import numpy as np

## linear regression
# Closed form
class LR:
    def __init__(self):
        self.w = None
    
    def fit(self, X, y):
        n = X.shape[0]
        X = np.concatenate([np.ones((n, 1)), X], axis=1)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X):
        n = X.shape[0]
        X = np.hstack([np.ones((n, 1)), X])
        return X @ self.w



# Gradient descent based
def loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def loss_vectorized(X, y, theta):
    m = len(y)
    cost = (1 / (2 * m)) * (X @ theta - y).T @ (X @ theta - y)
    return cost.item()

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = X @ theta
        error = X.T @ (predictions - y)
        descent = alpha * (1 / m) * error
        theta -= descent
        J_history[i] = loss_vectorized(X, y, theta)

    return theta, J_history
