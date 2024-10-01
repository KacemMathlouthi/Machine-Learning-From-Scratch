import numpy as np

class LinearRegression:
    """
    A Linear Regression model using gradient descent.

    Parameters:
    learning_rate (float): The learning rate for gradient descent. Default is 0.001.
    n_iters (int): The number of iterations for the gradient descent algorithm. Default is 1000.
    """

    def __init__(self, learning_rate=0.001, n_iters=1000) -> None:
        """Initialize the linear regression model.
        
        Args:
            learning_rate float: The learning rate for gradient descent optimization. Default is 0.001.
            n_iters int: The number of iterations for the optimization process. Default is 1000.
        
        Returns:
            None: This method initializes the model parameters and doesn't return anything.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        X (numpy.ndarray): The input feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): The target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))      # Gradient for weights
            db = (2/n_samples) * np.sum(y_pred - y)             # Gradient for bias

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict target values for the input data.

        Parameters:
        X (numpy.ndarray): The input feature matrix of shape (n_samples, n_features).

        Returns:
        numpy.ndarray: The predicted target values of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias
