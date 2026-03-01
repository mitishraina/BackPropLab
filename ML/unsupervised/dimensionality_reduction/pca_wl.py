# PCA or Prinicipal Component Analysis
# 1. unsupervised learning algo to reduce dataset dimensionality while retaining maximum variance(information)
# 2. transforms correlated features into a smaller set of uncorrelated "prinicipal components"
# 3. faster computation, noise reduction, and improved visualization and mitigates the "curse of dimensionality"

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        
        self.n_components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )
        
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    
# pca using SVD (numerically more stable)

class PCASVD:
    def __init__(self, n_components):
        self.n_components = n_components
        
        self.n_components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.components_ = Vt[:self.n_components].T
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        total_variance = np.sum(explained_variance)
        
        self.explained_variance_ratio_ = (
            explained_variance[:self.n_components] / total_variance
        )
    
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
# example usage
# from sklearn.datasets import load_digits
# import matplotlib.pyplot as plt

# data = load_digits()
# X = data.data
# y = data.target

# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X)

# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
# plt.title("PCA Projection (2D)")
# plt.show()

# print("Explained variance ratio:",
#       pca.explained_variance_ratio_)