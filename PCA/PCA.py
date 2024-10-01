import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation.

    This class performs PCA on a given dataset, reducing its dimensionality
    while preserving the most important features.

    Attributes:
        n_components (int): The number of principal components to keep.
        components (ndarray): The principal components (eigenvectors).
        mean (ndarray): The mean of the input data.

    """

    def __init__(self, n_components):
        """
        Initialize the PCA object.

        Args:
            n_components (int): The number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model to the input data.

        This method computes the principal components of the input data.

        Args:
            X (ndarray): Input data of shape (n_samples, n_features).

        """
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # Covariance
        cov = np.cov(X.T)

        # Eigen vectors and values
        eigenvectors, eigenvalues = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        """
        Apply dimensionality reduction to the input data.

        This method projects the input data onto the principal components.

        Args:
            X (ndarray): Input data of shape (n_samples, n_features).

        Returns:
            ndarray: Transformed data of shape (n_samples, n_components).
        """
        # Projecting the data
        X = X - self.mean
        return np.dot(X, self.components.T)