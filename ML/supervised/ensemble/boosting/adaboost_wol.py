# adaboost or adaptive boosting
# 1. supervised ensemble learning method(boosting technique or bias reduction technique)
# 2. combines predictions from multiple weak learners to create a powerful strong learner

import numpy as np

class Decision:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1
        
    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')
        
        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[polarity * feature_values < polarity * threshold] = -1
                    
                    misclassified = predictions != y
                    error = np.sum(sample_weights[misclassified])
                    
                    if error < min_error:
                        min_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity
                        
        def predict(self, X):
            n_samples = X.shape[0]
            predictions = np.ones(n_samples)
            feature_values = X[:, self.feature]
            predictions[
                self.polarity * feature_values < self.polarity * self.threshold
            ] = -1
            return predictions
        

class AdaBoost:
    """
    Binary Classification, lables must be -1 and 1
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            stump = Decision()
            stump.fit(X, y, w)
            
            predictions = stump.predict(X)
            error = np.sum(w[predictions != y])
            
            error = max(error, 1e-10)
            alpha = 0.5 * np.log((1- error) / error)
            
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            
            self.models.append(stump)
            self.alphas.append(alpha)
            
    def predict(self, X):
        model_preds = np.array(
            [alpha * model.predict(X)
             for model, alpha in zip(self.models, self.alphas)]
        )
        
        y_pred = np.sum(model_preds, axis=0)
        return np.sign(y_pred)