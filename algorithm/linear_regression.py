import numpy as np

## linear regression
class LR:
    def __init__(self):
        self.w = None
    
    def fit(self, X):
        n = X.shape[0]
        X = np.hstack([np.ones((n, 1)), X])
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X):
        n = X.shape[0]
        X = np.hstack([np.ones((n, 1)), X])
        return X @ self.w