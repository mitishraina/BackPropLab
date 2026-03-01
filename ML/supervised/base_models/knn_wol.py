# KNN or K-Nearest Neighbors
# 1. Supervised learning algorithm used for both classification and regression tasks
# 2. Based on the principle of "similarity" - it classifies a data point based on the majority class among its k nearest neighbors in the feature space
# For Classification: The new data point is assigned to the class that appears most frequently among its K nearest neighbors (a majority vote).
# For Regression: The algorithm predicts the value for the new point by taking the average (mean) of the values of its K nearest neighbors. 

import numpy as np
from collections import Counter

class KNN:
    """
    k: number of neighbors
    distance: 'euclidean' or 'manhattan'
    """
    
    def __init__(self, k=3, distance='euclidean'):
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(x1 - x2) ** 2)
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _compute_distance(self, x1, x2):
        if self.distance == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance == 'manhattan':
            return self._manhattan_distance(x1, x2)
        else:
            raise ValueError('unsupported distance metrics')
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        distances = [
            self._compute_distance(x, x_train)
            for x_train in self.X_train
        ]
        
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    

# example usage
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# def main():
#     data = load_iris()
#     X = data.data
#     y = data.target

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Important: scale features
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = KNN(k=5)
#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)

#     accuracy = np.mean(predictions == y_test)
#     print(f"Accuracy: {accuracy:.4f}")


# if __name__ == "__main__":
#     main()