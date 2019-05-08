import unittest
from src.model.baseline import BaseLine


class TestBaseLine(unittest.TestCase):
    def test_MFCC(self):
        classifier = BaseLine('SVM', 'MFCC', test=True)
        classifier.run()

    # def test_eGeMAPS(self):
    #     classifier = BaseLine('SVM', 'eGeMAPS', test=True)
    #     classifier.run()
    
    # def test_DeepSpectrum(self):
    #     classifier = BaseLine('SVM', 'Deep', test=True)
    #     classifier.run()
    
    # def test_BoAW(self):
    #     classifier = BaseLine('SVM', 'BoAW', test=True)
    #     classifier.run()
    
    def test_AU(self):
        classifier = BaseLine('SVM', 'AU', test=True)
        classifier.run()

    # def test_BoVW(self):
    #     classifier = BaseLine('SVM', 'BoVW', test=True)
    #     classifier.run()


if __name__ == '__main__':
    unittest.main()