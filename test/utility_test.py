import unittest
from src.utility import load_label, load_LLD, load_baseline_feature
from src.utility import load_MATLAB_baseline_feature


class TestUtility(unittest.TestCase):
    def test_load_label(self):
        label_train, label_dev = load_label()
    
    def test_load_LLD(self):
        load_LLD('MFCC', 'train', 1, verbose=True)
        load_LLD('eGeMAPS', 'dev', 30, verbose=True)
        load_LLD('openFace', 'test', 10, verbose=True)

    def test_load_baseline_feature(self):
        load_baseline_feature('BoAW', 'train', 100, verbose=True)
        load_baseline_feature('eGeMAPS', 'dev', 50, verbose=True)
        load_baseline_feature('MFCC', 'train', 10, verbose=True)
        load_baseline_feature('DeepSpectrum', 'dev', 4, verbose=True)
        load_baseline_feature('BoVW', 'test', 30, verbose=True)
        load_baseline_feature('AU', 'dev', 60, verbose=True)

    def test_load_MATLAB_baseline_feature(self):
        # load_MATLAB_baseline_feature('AU', verbose=True)
        # load_MATLAB_baseline_feature('BoW', verbose=True)
        load_MATLAB_baseline_feature('Deep', verbose=True)
        # load_MATLAB_baseline_feature('eGeMAPS', verbose=True)
        load_MATLAB_baseline_feature('MFCC', verbose=True)

    def test_load_MATLAB_label(self):
        # load_MATLAB_label('AU', verbose=True)
        # load_MATLAB_label('BoW', verbose=True)
        load_MATLAB_label('Deep', verbose=True)
        # load_MATLAB_label('eGeMAPS', verbose=True)
        load_MATLAB_label('MFCC', verbose=True)



if __name__ == "__main__":
    unittest.main()