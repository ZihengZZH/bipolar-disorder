import unittest
from src.model.random_forest import RandomForest


class TestRandomForest(unittest.TestCase):
    def test_random_forest(self):
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

        random_forest = RandomForest('IRIS', X_train, y_train, X_test, y_test, test=True)
        random_forest.run()


if __name__ == "__main__":
    unittest.main()