import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.2,
    random_state=42
)
# 1. without pipeline
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

labels = kmeans.predict(X_scaled)
centroids = kmeans.cluster_centers_

# 2. with pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, random_state=42))
])

pipeline.fit(X)

labels = pipeline.named_steps['kmeans'].labels_

centroids = pipeline.named_steps['kmeans'].cluster_centers_
X_scaled = pipeline.named_steps['scaler'].transform(X)


plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
plt.title("K-Means Clustering (Without Pipeline)")
plt.show()