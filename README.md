# Machine Learning From Scratch

## Overview
This project is an implementation of various popular machine learning algorithms built entirely from scratch using Python. Each algorithm is implemented in its own file (e.g., `DecisionTrees.py`) with a corresponding Jupyter notebook (`train.ipynb`) for testing and training on sample datasets.

### Implemented Algorithms:
1. **[Decision Tree](./Decision%20Tree/)** - A tree-based model used for classification and regression tasks.
2. **[K-Means Clustering](./K-Means/)** - A popular unsupervised learning algorithm used for clustering tasks.
3. **[K-Nearest Neighbors (KNN)](./KNN/)** - A simple, instance-based learning algorithm used for classification and regression.
4. **[Linear Regression](./Linear%20Regression/)** - A linear approach to modeling the relationship between a dependent variable and one or more independent variables.
5. **[Logistic Regression](./Logistic%20Regression/)** - A statistical method for binary classification tasks.
6. **[Naive Bayes](./Naive%20Bayes/)** - A probabilistic classifier based on Bayes' theorem.
7. **[Principal Component Analysis (PCA)](./PCA/)** - A dimensionality reduction technique.
8. **[Perceptron](./Perceptron/)** - One of the simplest types of artificial neural networks used for binary classification.
9. **[Random Forest](./Random%20Forest/)** - An ensemble learning method based on decision trees for classification and regression tasks.

### Features
- Each algorithm is implemented without relying on high-level machine learning libraries, focusing on understanding the fundamental principles behind each technique.
- A `train.ipynb` notebook is provided to demonstrate how to train and test each algorithm with a sample dataset.
  
### Getting Started

#### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required dependencies are listed in `requirements.txt`.

#### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/KacemMathlouthi/Machine-Learning-From-Scratch.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Machine-Learning-From-Scratch
    ```
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

#### Running the Notebooks
1. Open the `train.ipynb` notebook in Jupyter, e.g. the KNN training notebook:
    ```bash
    jupyter notebook KNN/train.ipynb
    ```
2. Follow the steps to test and train each algorithm on sample datasets.

### Configuration
- **`.gitignore`**: Lists files and directories to ignore in version control.
- **`requirements.txt`**: Contains a list of required Python packages.
- **`README.md`**: This file, providing an overview and usage guide for the project.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
