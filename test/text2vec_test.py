import unittest
from src.model.text2vec import Text2Vec
from src.model.random_forest import RandomForest


class TestText2Vec(unittest.TestCase):
    def test_text2vec(self):
        sample = Text2Vec()
        sample.build_model()
        # sample.train_model()
        # sample.infer_embedding('train')
        # sample.infer_embedding('dev')
        X_train, y_train = sample.load_embedding('train')
        X_dev, y_dev = sample.load_embedding('dev')
        random_forest = RandomForest('text', X_train, y_train, X_dev, y_dev, test=True)
        random_forest.run()
        random_forest.evaluate()


if __name__ == "__main__":
    unittest.main()