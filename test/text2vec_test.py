import unittest
from src.model.text2vec import Text2Vec
from src.model.random_forest import RandomForest


class TestText2Vec(unittest.TestCase):
    def test_text2vec(self):
        sample = Text2Vec(build_on_corpus=False)
        sample.build_model()
        sample.train_model()
        sample.infer_embedding('train')
        sample.infer_embedding('dev')
        sample.load_model()
        X_train, y_train = sample.load_embedding('train')
        X_dev, y_dev = sample.load_embedding('dev')
        random_forest = RandomForest('text', X_train, y_train, X_dev, y_dev, test=True)
        random_forest.run()
        random_forest.evaluate()
        sample.evaluate_model()
    
    def test_most_similar(self):
        text2vec = Text2Vec()
        text2vec.dm = 1
        text2vec.vector_size = 50
        text2vec.window_size = 10
        text2vec.negative = 5
        text2vec.hs = 0
        text2vec.build_model()
        text2vec.load_model()
        text2vec.evaluate_model()
        # text2vec.process_metadata_tensorboard()


if __name__ == "__main__":
    unittest.main()