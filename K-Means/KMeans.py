import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """
    K-Means clustering algorithm implementation.

    Parameters:
    -----------
    K : int, default=5
        The number of clusters to form.
    max_iters : int, default=100
        Maximum number of iterations for the algorithm.
    plot_steps : bool, default=False
        If True, plots the clustering process at each step.

    Attributes:
    -----------
    clusters : list
        List of clusters, where each cluster is a list of sample indices.
    centroids : list
        List of centroids for each cluster.
    """

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        """
        Perform K-Means clustering on the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.

        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels for each sample.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # Initialize centroids randomly
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_steps:
                self.plot()
            
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            if self._is_converged(centroids_old, self.centroids):
                break
            
            if self.plot_steps:
                self.plot()
        
        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        """
        Assign each sample to the nearest centroid.

        Parameters:
        -----------
        centroids : list
            List of current centroids.

        Returns:
        --------
        clusters : list
            List of clusters, where each cluster is a list of sample indices.
        """
        clusters = [[] for _ in range(self.K)]
        for index, sample in enumerate(self.X):
            centroid_index = self._closest_centroid(centroids, sample)
            clusters[centroid_index].append(index)
        return clusters
    
    def _closest_centroid(self, centroids, sample):
        """
        Find the index of the closest centroid to the given sample.

        Parameters:
        -----------
        centroids : list
            List of centroids.
        sample : array-like
            A single sample from the dataset.

        Returns:
        --------
        closest_idx : int
            Index of the closest centroid.
        """
        distances = [self._euclidean_distance(sample, centroid) for centroid in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
        -----------
        x1, x2 : array-like
            The two points to calculate the distance between.

        Returns:
        --------
        distance : float
            The Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2)**2))

    def _get_centroids(self, clusters):
        """
        Calculate new centroids based on the current clusters.

        Parameters:
        -----------
        clusters : list
            List of clusters, where each cluster is a list of sample indices.

        Returns:
        --------
        centroids : array, shape (K, n_features)
            New centroids for each cluster.
        """
        centroids = np.zeros((self.K, self.n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_index] = cluster_mean
        return centroids
    
    def _is_converged(self, centroids_old, centroids):
        """
        Check if the algorithm has converged.

        Parameters:
        -----------
        centroids_old : list
            List of old centroids.
        centroids : list
            List of new centroids.

        Returns:
        --------
        bool
            True if converged, False otherwise.
        """
        distances = [self._euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self, title=None):
        """
        Plot the current state of the clustering.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot samples
        scatter = ax.scatter(self.X[:, 0], self.X[:, 1], c=self._get_cluster_labels(self.clusters), 
                             cmap='viridis', alpha=0.7, s=40)
        
        # Plot centroids
        ax.scatter(np.array(self.centroids)[:, 0], np.array(self.centroids)[:, 1], 
                   c='red', s=200, alpha=0.8, marker='*', edgecolor='black', linewidth=1.5)
        
        # Customize the plot
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title or f'KMeans Clustering (K={self.K})', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def _get_cluster_labels(self, clusters):
        """
        Convert clusters to labels for each sample.

        Parameters:
        -----------
        clusters : list
            List of clusters, where each cluster is a list of sample indices.

        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels for each sample.
        """
        labels = np.empty(self.n_samples)
        for cluster_index, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_index
        return labels