#gradient boosting
# 1. ensemble method that builds a strong predictive model by sequentially combining multiple weak learners, typically decision trees
# 2. each new tree is trained to correct the errors made by the previous trees, allowing the model to focus on difficult-to-predict instances

import numpy as np

class GBNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value =value
        

class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = np.mean(y)
            return GBNode(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return GBNode(value=np.mean(y))
        
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return GBNode(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y):
        best_mse = float("inf")
        split_idx, split_threshold = None, None
        
        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                
                mse = (
                    self._mse(y[left_idx]) * len(y[left_idx]) +
                    self._mse(y[right_idx]) * len(y[right_idx])
                ) / n_samples
                
                if mse < best_mse:
                    best_mse = mse
                    split_idx = feature
                    split_threshold = threshold
                    
        return split_idx, split_threshold
    
    def _mse(self, y):
        return np.mean((y - np.meany(y)) ** 2)
    
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])
    
    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
    
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, lr=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        
        self.trees = []
        self.init_prediction = None
        
    def fit(self, X, y):
        self.init_prediction = np.mean(y)
        y_pred = np.full_like(y, self.init_prediction, dtype=float)
        
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            
            y_pred += self.lr * update
            self.trees.append(tree)
            
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_prediction)
        
        for tree in self.trees:
            y_pred += self.lr * tree.predict(X)
            
        return y_pred
    
# example usage
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# import numpy as np

# X, y = make_regression(
#     n_samples=1000,
#     n_features=5,
#     noise=20,
#     random_state=42
# )

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = GradientBoostingRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3
# )

# model.fit(X_train, y_train)

# preds = model.predict(X_test)

# mse = np.mean((preds - y_test) ** 2)
# print("MSE:", mse)