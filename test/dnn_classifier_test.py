import unittest
from src.model.dnn_classifier import SingleTaskDNN, MultiTaskDNN
from src.utils.io import load_proc_baseline_feature


class TestDNNClassifier(unittest.TestCase):
    def test_single_task_dnn(self):
        X_train, y_train, _, X_dev, y_dev, _ = load_proc_baseline_feature('AU', verbose=True)
        assert X_train.shape[1] == X_dev.shape[1]
        test_dnn = SingleTaskDNN('AU', X_train.shape[1])
        test_dnn.build_model()
        test_dnn.train_model(X_train, y_train, X_dev, y_dev)
        test_dnn.evaluate_model(X_dev, y_dev)


if __name__ == '__main__':
    unittest.main()