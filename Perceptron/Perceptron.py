import numpy as np

def unit_step_func(x):
    """
    Unit step function.
    
    Args:
    x (float or array): Input value(s)
    
    Returns:
    1 if x > 0, else 0
    """
    return np.where(x > 0, 1, 0)

class Perceptron:
    """
    Perceptron classifier.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for weight updates.
    n_iters : int, default=1000
        Number of iterations over the training dataset.

    Attributes:
    -----------
    lr : float
        Learning rate.
    n_iters : int
        Number of iterations.
    activation_func : function
        Activation function to use.
    weights : array
        Weights after fitting.
    bias : float
        Bias after fitting.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initializes the Perceptron model with specified learning rate and number of iterations.
        
        Args:
            learning_rate (float): The learning rate for weight updates. Default is 0.01.
            n_iters (int): The number of iterations for training. Default is 1000.
        
        Returns:
            None
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the perceptron to the training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features) Training vectors
        y : array-like, shape (n_samples,) Target values.
        """
        n_samples, n_features = X.shape
        
        # Initialize weights randomly
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        y = unit_step_func(y)

        # Learning Weights
        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)

                # Updating weights and bias
                update = self.lr * (y[index] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y : array, shape (n_samples,)
            The predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred