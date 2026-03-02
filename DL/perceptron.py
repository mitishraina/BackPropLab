# Frank Rosenblatt introduced perceptron in 1957
# The first trainable, single layer ANN 
# limited to linearly separable and binary output problems
# classifier data by multiplying inputs by learned weights, summing them and applying a threshold

import numpy as np

class Perceptron:
    """
    Perceptron labels must be -1 or 1
    """
    
    def __init__(self, lr=0.01, n_iters=1000, seed=42):
        self.lr = lr
        self.n_iters = n_iters
        self.seed = seed
        
        self.weights = None
        self.bias = None
        self.error_ = []
        
    def fit(self, X, y):
        np.random.seed(self.seed)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.n_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                predictions = self._activation(linear_output)
                
                if y[idx] != predictions:
                    update = self.lr * y[idx]
                    self.weights += update * x_i
                    self.bias += update
                    
                    errors += 1
                self.error_.append(errors)
                
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)
    
    def _activation(self, z):
        return np.where(z >= 0, 1, -1)
    
    
# Perceptron works but fails because of no hidden layers, no differentiable activation, cannot learn nonlinear decisions
# and this led to MLP & Backpropogation