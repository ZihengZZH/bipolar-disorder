import unittest
import numpy as np
import pandas as pd
from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.model.autoencoder_multimodal import AutoEncoderMultimodal
from src.utils.io import load_bags_of_words
from src.utils.io import load_proc_baseline_feature


class TestAutoEncoder(unittest.TestCase):
    def test_autoencoder(self):
        X_train_V, X_dev_V, X_test_V, y_train_V, inst_train_V, y_dev_V, inst_dev_V = load_bags_of_words('BoVW', verbose=True)

        ae_bovw = AutoEncoder('BoVW', X_train_V.shape[1])
        ae_bovw.build_model()
        
        ae_bovw.train_model(pd.concat([X_train_V, X_dev_V]), X_test_V)
        ae_bovw.encode(X_train_V, X_dev_V)
        encoded_train, encoded_dev = ae_bovw.load_presentation()
    
    def test_bimodal_autoencoder(self):
        X_train_A, X_dev_A, X_test_A, y_train_A, inst_train_A, y_dev_A, inst_dev_A = load_bags_of_words('BoAW', verbose=True)
        X_train_V, X_dev_V, X_test_V, y_train_V, inst_train_V, y_dev_V, inst_dev_V = load_bags_of_words('BoVW', verbose=True)

        bae = AutoEncoderBimodal('bimodal_BoXW', X_train_A.shape[1], X_train_V.shape[1])
        bae.build_model()

        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()

    def test_multimodal_autoencoder(self):
        ae = AutoEncoder('test', 118)
        ae.build_model()
        bae = AutoEncoderBimodal('test', 120, 118)
        bae.build_model()
        mae = AutoEncoderMultimodal('test', 118, 136, 6, 6, 35)
        mae.build_model()

if __name__ == "__main__":
    unittest.main()