import unittest
from src.utility.io import load_label, load_LLD, load_baseline_feature
from src.utility.io import load_proc_baseline_feature
from src.utility.io import load_audio_file
from src.utility.preprocess import preproc_baseline_feature
from src.utility.preprocess import preproc_transcript


class TestUtility(unittest.TestCase):
    def test_load_label(self):
        load_label(verbose=True)
    
    def test_load_LLD(self):
        load_LLD('MFCC', 'train', 1, verbose=True)
        load_LLD('eGeMAPS', 'dev', 30, verbose=True)
        load_LLD('openFace', 'test', 10, verbose=True)

    def test_load_baseline_feature(self):
        load_baseline_feature('BoAW', 'train', 100, verbose=True)
        load_baseline_feature('eGeMAPS', 'dev', 50, verbose=True)
        load_baseline_feature('MFCC', 'train', 10, verbose=True)
        load_baseline_feature('Deep', 'dev', 4, verbose=True)
        load_baseline_feature('BoVW', 'test', 30, verbose=True)
        load_baseline_feature('AU', 'dev', 60, verbose=True)

    def test_load_proc_baseline_feature(self):
        load_proc_baseline_feature('Deep', verbose=True)
        load_proc_baseline_feature('eGeMAPS', verbose=True)
        load_proc_baseline_feature('MFCC', verbose=True)
        load_proc_baseline_feature('BoAW', verbose=True)
        load_proc_baseline_feature('BoVW', verbose=True)
        load_proc_baseline_feature('AU', verbose=True)
    
    def test_load_audio_file(self):
        load_audio_file(None, None, verbose=True)
        load_audio_file('dev', 23, verbose=True)

    def test_preprocess_transcript(self):
        preproc_transcript('all')


if __name__ == "__main__":
    unittest.main()