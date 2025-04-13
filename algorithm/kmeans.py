import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# --- KMeans Implementation ---
class KMeans:
    def __init__(self, k=4, max_iters=200, tol=1e-4):
        self.k = k 
        self.max_iters = max_iters 
        self.tol = tol 
    
    def fit(self, X):
        indices = np.random.choice(len(X), self.k, replace=False)
        centroids = X[indices]

        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]

            for x in X:
                dists = [np.linalg.norm(x - centroid) for centroid in centroids]
                closest_cluster_idx = np.argmin(dists)
                clusters[closest_cluster_idx].append(x) 
            
            new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i]
                                      for i, cluster in enumerate(clusters)])
            
            if np.allclose(new_centroids, centroids, atol=self.tol):
                break
            centroids = new_centroids

        self.centroids = centroids
        self.clusters = clusters 
    
    def predict(self, X):
        return np.array([
            np.argmin([np.linalg.norm(x - centroid) for centroid in self.centroids])
            for x in X
        ])

# Create and fit KMeans
kmeans = KMeans(k=4)
kmeans.fit(X)

# Predict cluster labels
y_pred = kmeans.predict(X)

# Print centroids
print("Centroids:")
print(kmeans.centroids)

# Plot the result
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.75)
plt.show()
