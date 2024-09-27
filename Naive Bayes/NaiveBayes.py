import numpy as np

class NaiveBayes:
    """
    Gaussian Naive Bayes classifier
    """

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.

        Args:
            X (np.array): Training data of shape (n_samples, n_features).
            y (np.array): Target values of shape (n_samples,).

        Returns:
            None
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Args:
            X (np.array): Data to predict, of shape (n_samples, n_features).

        Returns:
            np.array: Predicted class labels, of shape (n_samples,).
        """
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        """
        Predict the class label for a single sample.

        Args:
            x (np.array): Single sample to predict, of shape (n_features,).

        Returns:
            Any: Predicted class label.
        """
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._probability_density(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _probability_density(self, class_idx, x):
        """
        Calculate the probability density function for a given class and sample.

        This method implements the Gaussian probability density function.

        Args:
            class_idx (int): Index of the class.
            x (np.array): Sample to calculate the probability for.

        Returns:
            np.array: Probability density for each feature.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator