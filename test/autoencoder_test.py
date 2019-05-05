import unittest
from src.autoencoder import AutoEncoder


class TestAutoEncoder(unittest.TestCase):
    def test_build_model(self):
        ae = AutoEncoder('MFCC')
        ae.build_model()
        ae.load_model()
    

if __name__ == "__main__":
    unittest.main()