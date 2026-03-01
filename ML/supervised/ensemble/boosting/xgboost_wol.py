# xgboost (Extreme gradient boosting)
# 1. highly efficient and scalable implementation of gradient boosting that incorporates regularization to prevent overfitting, making it a popular choice for structured data tasks
# 2. it uses a more sophisticated tree-building algorithm and supports parallel processing, which allows it to handle large datasets and achieve faster training times compared to traditional gradient boosting methods
# XGBoost is a second order gradient boosting algorithm that builds trees sequentially using gradients and hessians of the loss function, while applying regularization and smart split gain calculation to prevent overfitting and improve efficieny
# If the dataset has many categorical features, prefer CatBoost 
# For very large datasets where speed is critical, use LightGBM
# For stable, well-tested performance and fine control, use XGBoost

import numpy as np

class XGBNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class XGBTree:
    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.root = None

    def fit(self, X, gradients, hessians):
        self.root = self._grow_tree(X, gradients, hessians)

    def _grow_tree(self, X, g, h, depth=0):
        if depth >= self.max_depth or X.shape[0] == 0:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lambda_)
            return XGBNode(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, g, h)

        if best_feature is None:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lambda_)
            return XGBNode(value=leaf_value)

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        left = self._grow_tree(
            X[left_idx], g[left_idx], h[left_idx], depth + 1
        )
        right = self._grow_tree(
            X[right_idx], g[right_idx], h[right_idx], depth + 1
        )

        return XGBNode(best_feature, best_threshold, left, right)

    def _best_split(self, X, g, h):
        best_gain = -float("inf")
        split_idx, split_threshold = None, None

        n_samples, n_features = X.shape

        G = np.sum(g)
        H = np.sum(h)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                G_L = np.sum(g[left_idx])
                H_L = np.sum(h[left_idx])
                G_R = np.sum(g[right_idx])
                H_R = np.sum(h[right_idx])

                gain = 0.5 * (
                    (G_L ** 2) / (H_L + self.lambda_) +
                    (G_R ** 2) / (H_R + self.lambda_) -
                    (G ** 2) / (H + self.lambda_)
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
    
    
class XGBoostRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        lambda_=1.0,
        gamma=0.0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma

        self.trees = []
        self.base_score = None

    def fit(self, X, y):
        self.base_score = np.mean(y)
        y_pred = np.full_like(y, self.base_score, dtype=float)

        for _ in range(self.n_estimators):
            gradients = y_pred - y
            hessians = np.ones_like(y)

            tree = XGBTree(
                max_depth=self.max_depth,
                lambda_=self.lambda_,
                gamma=self.gamma
            )

            tree.fit(X, gradients, hessians)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_score)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

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

# model = XGBoostRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3,
#     lambda_=1.0,
#     gamma=0.1
# )

# model.fit(X_train, y_train)

# preds = model.predict(X_test)

# mse = np.mean((preds - y_test) ** 2)
# print("MSE:", mse)