import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent.

    Parameters:
    learning_rate : float, optional (default=0.001)
        The step size for the gradient descent optimization.
    n_iters : int, optional (default=1000)
        The number of iterations to run gradient descent.

    Attributes:
    weights : ndarray of shape (n_features,)
        Coefficients of the model.
    bias : float
        Bias term (intercept) of the model.
    """

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.

        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values (0 or 1).
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_hat = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        predictions : list
            Predicted class labels (0 or 1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_hat = sigmoid(linear_model)

        # Convert probabilities to binary class labels
        predictions = [1 if i > 0.5 else 0 for i in y_hat]
        return predictions
