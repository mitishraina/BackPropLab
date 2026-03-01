# k means
# unsupervised learning algorithm
# 1. partition data into k clusters based on feature similarity
# 2. iteratively assigns data points to the nearest cluster centroid and updates the centroids until convergence is achieved

import numpy as np

class KMeans:
    """
    KMeans clustering Parameters:
    n_clusters: int, default=8, Number of clusters
    max_iter: int, Maximum number of iterations
    tol: float, convergence tolerance
    seed: int, for reproducibility
    """
    
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        
        self.centroids = None
        self.labels_ = None
        self.intertia_ = None
        
    def _initialize_centrods(self, X):
        np.random.seed(self.seed)
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]
    
    def _compute_distance(self, X, centroids):
        return np.sqrt(
            np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)
        )
        
    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                centroids[k] = X[np.random.randint(0, X.shape[0])]
            else:
                centroids[k] = np.mean(cluster_points, axis=0)
                
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        total = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            total += np.sum((cluster_points - centroids[k]) ** 2)
            
        return total
    
    def fit(self, X):
        self.centroids = self._initialize_centrods(X)
        for _ in range(self.max_iters):
            distances = self._compute_distance(X, self.centroids)
            labels = self._assign_clusters(distances)
            
            new_centroids = self._update_centroids(X, labels)
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            
            if shift < self.tol:
                break
            
        self.labels_ = labels
        self.intertia_ = self._compute_inertia(X, labels, self.centroids)
        
    def predict(self, X):
        distances = self._compute_distance(X, self.centroids)
        return self._assign_clusters(distances)
    
    
# example usage
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# X, _ = make_blobs(
#     n_samples=500,
#     centers=3,
#     cluster_std=1.2,
#     random_state=42
# )

# model = KMeans(n_clusters=3)
# model.fit(X)

# labels = model.labels_

# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.scatter(
#     model.centroids[:, 0],
#     model.centroids[:, 1],
#     marker='x',
#     s=200
# )
# plt.show() 

# print("Inertia:", model.inertia_)