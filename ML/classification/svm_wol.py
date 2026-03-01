# SVM or support vector machine
# 1. supervised learning algorithm used for classification and regression tasks
# 2. finds the optimal hyperplane that best separates data points of different classes in a high-dimensional space


import numpy as np

class LinearSVM:
    def __init__(self, lr=0.001, lambda_=0.01, n_iters=1000, seed: int=42):
        self.lr = lr
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.seed = seed
        
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        np.random.seed(self.seed)
        
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b)
                
                if condition >= 1:
                    dw = 2 * self.lambda_ * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_ * self.w - y[idx] * x_i
                    db = -y[idx]
                    
                self.w -= self.lr * dw
                self.b -= self.lr * db
                
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

# example usage

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# X, y = make_classification(
#     n_samples=1000,
#     n_features=2,
#     n_informative=2,
#     n_redundant=0,
#     random_state=42
# )

# y = np.where(y == 0, -1, 1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = LinearSVM(
#     learning_rate=0.001,
#     lambda_=0.01,
#     n_iters=1000
# )

# model.fit(X_train, y_train)

# predictions = model.predict(X_test)

# accuracy = np.mean(predictions == y_test)
# print("Accuracy:", accuracy)