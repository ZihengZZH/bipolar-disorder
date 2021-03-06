import unittest
from src.model.text2vec import Text2Vec
from src.model.dnn_classifier import SingleTaskDNN, MultiTaskDNN
from src.utils.io import load_proc_baseline_feature, load_label


class TestDNNClassifier(unittest.TestCase):
    def test_single_task_dnn(self):
        X_train, y_train, _, X_dev, y_dev, _ = load_proc_baseline_feature('BoAW', verbose=True)
        assert X_train.shape[1] == X_dev.shape[1]
        num_classes = max(max(y_train), max(y_dev))
        test_dnn = SingleTaskDNN('BoAW', X_train.shape[1], num_classes)
        test_dnn.build_model()
        test_dnn.train_model(X_train, y_train, X_dev, y_dev)
        test_dnn.evaluate_model(X_dev, y_dev)

    def test_multi_task_dnn(self):
        X_train, y_train, inst_train, X_dev, y_dev, inst_dev = load_proc_baseline_feature('BoAW', verbose=True)
        ymrs_dev, ymrs_train, _, _ = load_label()

        self.assertEqual(X_train.shape[1], X_dev.shape[1])

        num_classes = max(max(y_train), max(y_dev))
        test_dnn = MultiTaskDNN('BoAW', X_train.shape[1], num_classes)

        y_dev_r = test_dnn.prepare_regression_label(ymrs_dev.values[:, 1], inst_dev)
        y_train_r = test_dnn.prepare_regression_label(ymrs_train.values[:, 1], inst_train)

        self.assertEqual(len(y_dev_r), len(y_dev))
        self.assertEqual(len(y_train_r), len(y_train))

        test_dnn.build_model()
        test_dnn.train_model(X_train, y_train, y_train_r, X_dev, y_dev, y_dev_r)
        test_dnn.evaluate_model(X_train, y_train, y_train_r, X_dev, y_dev, y_dev_r)


if __name__ == '__main__':
    unittest.main()