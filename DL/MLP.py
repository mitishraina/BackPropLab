# MLP or Multi Layer Perceptron
# single hidden layer MLP(binary classification)

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, lr: float=0.01, n_iters: int=1000, seed: int=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_iters = n_iters
        self.seed = seed
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        np.random.seed(self.seed)
        
        # initializes weights between input & hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        
        # initializes weights between hidden & output layer
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        
    def _relu(self, Z):
        return np.maximum(0, Z)
    
    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def _sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def _forward(self, X):
        # for input & hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._relu(self.Z1)
        
        # for hidden & output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._sigmoid(self.Z2)
        
        return self.A2
    
    def _compute_loss(self, y, y_hat):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            y * np.log(y_hat) + 
            (1 - y) * np.log(1 - y_hat)
        )
        
        return loss
    
    def _backward(self, X, y):
        m = X.shape[0]
        
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
    def fit(self, X, y):
        y = y.reshape(-1, 1)
        
        for _ in range(self.n_iters):
            y_hat = self._forward(X)
            loss = self._compute_loss(y, y_hat)
            self._backward(X, y)
            
    def predict(self, X):
        y_hat = self._forward(X)
        return (y_hat >= 0.5).astype(int)