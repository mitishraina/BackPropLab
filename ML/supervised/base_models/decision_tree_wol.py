# Decision Tree
# 1. Supervised learning algo that uses a flowchart like tree structure to model decisions
# 2. works for both classification and regression tasks
# 3. For classification: The tree is built by splitting the data based on feature values that maximize the separation of classes (e.g., using Gini impurity or information gain).
# 4. For regression: The tree is built by splitting the data based on feature values that minimize

import numpy as np
from collections import Counter

class Node:
    """
    Represents a single node in decision tree
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class DecisionTree:
    def __init__(self, max_depth: int=10, min_samples_split: int=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.criterion = criterion
        
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (
            depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return Node(value=self._most_common_label(y))
        
        left_idxs, right_idxs = self._split(
            X[:, best_feature], best_threshold
        )
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _split(self, feature_column, threshold):
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]
        
        return left_idxs, right_idxs
    
    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))
    
    def _information_gain(self, y, feature_column, threshold):
        parent_impurity = (
            self._gini(y) if self.self.criterion == "gini"
            else self._entropy(y)
        )
        
        left_idxs, right_idxs = self._split(feature_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        
        left_impurity = (
            self._gini(y[left_idxs]) if self.criterion == "gini"
            else self._entropy(y[left_idxs])
        )
        right_impurity = (
            self._gini(y[right_idxs]) if self.criterion == "gini"
            else self._entropy(y[right_idxs])
        )
        
        child_impurity = (
            (n_left / n) * left_impurity + (n_right / n) * right_impurity 
        )
        
        return parent_impurity - child_impurity
    
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    
    
    
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

# model = DecisionTree(max_depth=5)
# model.fit(X_train, y_train)

# predictions = model.predict(X_test)

# accuracy = np.mean(predictions == y_test)
# print("Accuracy:", accuracy)