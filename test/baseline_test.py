import unittest
from src.baseline import BaseLine


class TestBaseLine(unittest.TestCase):
    def test_MFCC(self):
        classifier = BaseLine('MFCC')
        classifier.run()

    def test_DeepSpectrum(self):
        classifier = BaseLine('Deep')
        classifier.run()

if __name__ == '__main__':
    unittest.main()