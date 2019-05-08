import unittest
from src.model.linear_svm import LinearSVM


class TestLinearSVM(unittest.TestCase):
    def test_Linear_SVM(self):
        import numpy as np
        from sklearn import datasets
        iris = datasets.load_iris()

        # load the iris dataset
        # NOTE that we only use first two features for 2-d plot
        X = iris.data[:,:2]
        y = iris.target
        indices = np.random.permutation(len(X))
        test_size = 15
        X_train = X[indices[:-test_size]]
        y_train = y[indices[:-test_size]]
        X_test = X[indices[-test_size:]]
        y_test = y[indices[-test_size:]]

        linear_svm = LinearSVM('IRIS', X_train, y_train, X_test, y_test, test=True)
        linear_svm.run()


if __name__ == "__main__":
    unittest.main()