import unittest
from src.utils.io import load_label, load_LLD, load_baseline_feature
from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_audio_file
from src.utils.io import load_aligned_features
from src.utils.preprocess import preproc_baseline_feature
from src.utils.preprocess import preproc_transcript
from src.utils.preprocess import process_corpus
from src.utils.preprocess import preprocess_AU
from src.utils.preprocess import preprocess_BOXW
from src.utils.preprocess import preprocess_align
from src.utils.preprocess import upsample
from src.utils.preprocess import frame2session
from src.utils.preprocess import k_fold_cv
from src.utils.preprocess import preprocess_reconstruction
from src.utils.vis import visualize_landmarks


class TestUtility(unittest.TestCase):
    def test_load_label(self):
        _, _, y_dev, y_train = load_label(verbose=True)
        y_dev = y_dev.values[:, 1]
        y_train = y_train.values[:, 1]

    def test_load_LLD(self):
        load_LLD('MFCC', 'train', 1, verbose=True)
        load_LLD('eGeMAPS', 'dev', 30, verbose=True)
        load_LLD('openFace', 'test', 10, verbose=True)

    def test_load_baseline_feature(self):
        load_baseline_feature('MFCC', 'train', 10, verbose=True)
        load_baseline_feature('eGeMAPS', 'dev', 50, verbose=True)
        load_baseline_feature('Deep', 'dev', 4, verbose=True)
        load_baseline_feature('BoAW', 'train', 50, verbose=True)
        load_baseline_feature('AU', 'dev', 60, verbose=True)
        load_baseline_feature('BoVW', 'train', 50, verbose=True)

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
    
    def test_process_corpus(self):
        process_corpus(verbose=True)

    def test_preprocess_AU(self):
        preprocess_AU(verbose=True)

    def test_visualize_landmarks(self):
        visualize_landmarks('train', index=10)

    def test_preprocess_BoXW(self):
        preprocess_BOXW(verbose=True)

    def test_preprocess_align(self):
        preprocess_align(eGeMAPS=True, verbose=True)

    def test_load_aligned_features(self):
        load_aligned_features(eGeMAPS=True, verbose=True)

    def test_upsample(self):
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('MFCC', verbose=True)
        X_train, y_train, train_inst = upsample(X_train, y_train, train_inst, verbose=True)
        print(X_train.shape, y_train.shape, train_inst.shape)
        from collections import Counter
        print(Counter(y_train))
    
    def test_frame2session(self):
        import numpy as np
        X = np.random.random((60, 30))
        y = np.ones((60, ))
        inst = np.hstack((np.ones(5,), np.ones(25,)*2, np.ones(30,)*3))
        X_sess, y_sess = frame2session(X, y, inst, verbose=True)
        print(type(X_sess), type(y_sess))

    def test_k_fold_cv(self):
        ids = k_fold_cv(20)
        for (ids_train, ids_dev) in ids:
            print(ids_train, ids_dev)

    def test_preprocess_reconstruction(self):
        preprocess_reconstruction()

if __name__ == "__main__":
    unittest.main()