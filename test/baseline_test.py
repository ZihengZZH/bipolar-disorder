import unittest
from src.baseline import BaseLine


class TestBaseLine(unittest.TestCase):
    def test_MFCC(self):
        classifier = BaseLine('RF', 'MFCC', test=True)
        classifier.run()

    def test_eGeMAPS(self):
        classifier = BaseLine('RF', 'eGeMAPS', test=True)
        classifier.run()
    
    def test_DeepSpectrum(self):
        classifier = BaseLine('RF', 'Deep', test=True)
        classifier.run()
    
    def test_BoAW(self):
        classifier = BaseLine('RF', 'BoAW', test=True)
        classifier.run()
    
    def test_AU(self):
        classifier = BaseLine('RF', 'AU', test=True)
        classifier.run()

    def test_BoVW(self):
        classifier = BaseLine('RF', 'BoVW', test=True)
        classifier.run()

if __name__ == '__main__':
    unittest.main()