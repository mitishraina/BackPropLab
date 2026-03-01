# Logistic Regression
# 1. supervised algorithm to predict binary outcomes based on one or more input features
# 2. models the probability of the default class (class 0) using the logistic function (sigmoid)
# f(x) = 1 / (1 + exp(-z)), where z = wx + b, w=weights, b=bias, x=input_features 

import numpy as np

class LogisticRegression:
    """
    lr or learning_rate: step size for gradient descent
    n_iters: number of iterations for training
    fit_intercept: whether to include bias term
    seed: random seed for reproducibility
    """
    def __init__(self, lr: float=0.01, n_iters: int=1000, fit_intercept: bool=True, seed: int=42):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.seed = seed
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _initialize_parameters(self, n_features: int):
        np.random.seed(self.seed)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1+np.exp(-z))
    
    def _compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def _compute_gradients(self, X, y_true, y_pred):
        m = X.shape[0]
        
        error = y_pred - y_true
        dw = (1/m) * np.dot(X.T, error)
        db = (1/m) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            dw, db = self._compute_gradients(X, y, y_pred)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_probability(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_probability(X)
        return (probabilities >= threshold).astype(int)


# usage example

# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split

# def main():
#     data = load_breast_cancer()
#     X = data.data
#     y = data.target

#     # Normalize features (important for convergence)
#     X = (X - X.mean(axis=0)) / X.std(axis=0)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = LogisticRegression(
#         learning_rate=0.01,
#         n_iters=2000
#     )

#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)

#     accuracy = np.mean(predictions == y_test)
#     print(f"Accuracy: {accuracy:.4f}")


# if __name__ == "__main__":
#     main()