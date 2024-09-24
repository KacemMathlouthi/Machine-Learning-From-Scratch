import numpy as np
from collections import Counter
from typing import List

class KNN:
    """
    K-Nearest Neighbors (KNN) classifier.

    Parameters:
    -----------
    k : int, default=3
        Number of nearest neighbors to consider for majority voting.
    """
    def __init__(self, k = 3):
        """
        Initializes the KNN classifier with the number of neighbors.

        Parameters:
        -----------
        k : int, default=3
            Number of neighbors to consider for the voting process.
        """
        if k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the model using the training data.

        Parameters:
        -----------
        X : np.ndarray
            Training data features of shape (n_samples, n_features).
        Y : np.ndarray
            Training data labels of shape (n_samples,).
        """
        self.X_train = X
        self.Y_train = Y

    def predict(self, X: np.ndarray) -> List:
        """
        Predict the class labels for the given test data.

        Parameters:
        -----------
        X : np.ndarray
            Test data of shape (n_samples, n_features).

        Returns:
        --------
        List
            Predicted labels for each test data point.
        """
        return [self._predict(x) for x in X]

    def _predict(self, x: np.ndarray) -> int:
        """
        Helper function to predict the class label for a single test point.

        Parameters:
        -----------
        x : np.ndarray
            A single test data point of shape (n_features,).

        Returns:
        --------
        int
            Predicted class label.
        """
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #Get the closest K neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        #Majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
        

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two points.

    Parameters:
    -----------
    x1 : np.ndarray
        First point of shape (n_features,).
    x2 : np.ndarray
        Second point of shape (n_features,).

    Returns:
    --------
    float
        Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1-x2)**2))
