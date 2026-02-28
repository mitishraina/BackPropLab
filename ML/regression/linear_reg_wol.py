# Linear Regression
# 1. supervised algorithm to predict continuous values based on one or more input features
# 2. finds the best-fit line or hyperplane in higher dimensions that minimizes the difference between predicted and actual values
# f(x) = wx+b, where w=weights, b=bias, x=input_features

import numpy as np

class LinearRegression:
    """act
    Linear Regression using gradient descent
    """
    def __init__(self, lr: float=0.01, n_iters: int=1000, fit_intercept: bool=True, seed: int=42):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.seed = seed
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _initialize(self, n_features):
        np.random.seed(self.seed)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
    def _compute_loss(self, y, y_pred):
        return np.mean((y-y_pred)**2)
    
    def _compute_gradients(self, X, y, y_pred):
        m = X.shape[0]
        error = y_pred - y
        
        dw = (2/m) * np.dot(X.T, error)
        db = (2/m) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize(n_features)
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            dw, db = self._compute_gradients(X, y, y_pred)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    
# example usage
# import numpy as np
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# def main():
#     X, y = make_regression(
#         n_samples=1000,
#         n_features=5,
#         noise=10,
#         random_state=42
#     )

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = LinearRegression(
#         learning_rate=0.01,
#         n_iters=2000
#     )

#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)

#     mse = np.mean((predictions - y_test) ** 2)
#     print(f"MSE: {mse:.4f}")


# if __name__ == "__main__":
#     main()