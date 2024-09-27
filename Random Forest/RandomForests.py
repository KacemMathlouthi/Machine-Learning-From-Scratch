import numpy as np
from collections import Counter
from DecisionTree import DecisionTree

class RandomForests:
    """
    Random Forests classifier.

    This class implements the Random Forests algorithm, an ensemble learning
    method that operates by constructing multiple decision trees during
    training and outputting the class that is the mode of the classes
    predicted by individual trees.

    Attributes:
        n_trees (int): The number of trees in the forest.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        max_depth (int): The maximum depth of the tree.
        n_features (int): The number of features to consider when looking for the best split.
        trees (list): List of DecisionTree objects.
    """

    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10, n_features=None):
        """
        Initialize the RandomForests object.

        Args:
            n_trees (int): The number of trees in the forest.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            max_depth (int): The maximum depth of the tree.
            n_features (int): The number of features to consider when looking for the best split.
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Args:
            X (np.array): The input samples.
            y (np.array): The target values.
        """
        for _ in range(self.n_trees):
            clf = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            X_samples, y_samples = self._bootstrap_samples(X, y)
            clf.fit(X_samples, y_samples)
            self.trees.append(clf)

    def _bootstrap_samples(self, X, y):
        """
        Create a bootstrap sample from the dataset.

        Args:
            X (np.array): The input samples.
            y (np.array): The target values.

        Returns:
            tuple: A tuple containing the bootstrap samples for X and y.
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        """
        Find the most common label in a dataset.

        Args:
            y (np.array): The target values.

        Returns:
            Any: The most common label.
        """
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        """
        Predict class for X.

        Args:
            X (np.array): The input samples.

        Returns:
            np.array: The predicted class labels for samples in X.
        """
        predictions = np.array([self.trees[i].predict(X) for i in range(self.n_trees)])
        predictions = np.swapaxes(predictions, 0, 1)
        return [self._most_common_label(prediction) for prediction in predictions]