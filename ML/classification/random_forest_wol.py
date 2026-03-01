# random forest
# 1. supervised learning algo or bagging ensemble learning method(variance reduction technique)
# 2. builds multiple decision tress during training and merges their predictions to produce a more accurate and stable result that a single tree

import numpy as np
from collections import Counter

class RFNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class RandomForest:
    def __init__(
        self,
        n_estimators=10,
        max_depth=10,
        min_samples_split=2,
        max_features=None,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = self._build_tree(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array(
            [self._predict_tree(tree, X) for tree in self.trees]
        )

        # Majority vote across rows
        predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            predictions.append(self._most_common_label(votes))

        return np.array(predictions)
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return RFNode(value=leaf_value)

        feature_indices = self._select_features(n_features)

        best_feature, best_threshold = self._best_split(
            X, y, feature_indices
        )

        if best_feature is None:
            return RFNode(value=self._most_common_label(y))

        left_idx, right_idx = self._split(
            X[:, best_feature], best_threshold
        )

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return RFNode(best_feature, best_threshold, left, right)

    def _select_features(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        return np.random.choice(
            n_features,
            self.max_features,
            replace=False
        )

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                gain = self._information_gain(
                    y, X[:, feature], threshold
                )

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    def _split(self, feature_column, threshold):
        left = np.where(feature_column <= threshold)[0]
        right = np.where(feature_column > threshold)[0]
        return left, right


    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _information_gain(self, y, feature_column, threshold):
        parent_impurity = self._gini(y)

        left_idx, right_idx = self._split(feature_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)

        left_impurity = self._gini(y[left_idx])
        right_impurity = self._gini(y[right_idx])

        child_impurity = (
            (n_left / n) * left_impurity +
            (n_right / n) * right_impurity
        )

        return parent_impurity - child_impurity


    def _predict_tree(self, tree, X):
        return np.array([self._traverse_tree(x, tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    
# example usage
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# import numpy as np


# data = load_iris()
# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = RandomForest(
#     n_estimators=20,
#     max_depth=8,
#     max_features=2
# )

# model.fit(X_train, y_train)

# predictions = model.predict(X_test)

# accuracy = np.mean(predictions == y_test)
# print("Accuracy:", accuracy)