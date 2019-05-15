import numpy as np
import unittest
from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_lstm import AutoEncoderLSTM
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.utils.io import load_proc_baseline_feature


class TestAutoEncoder(unittest.TestCase):
    def test_build_model(self):
        X_train_1, y_train_1, train_inst_1, X_dev_1, y_dev_1, dev_inst_1 = load_proc_baseline_feature('BoAW', verbose=True)
        X_train_2, y_train_2, train_inst_2, X_dev_2, y_dev_2, dev_inst_2 = load_proc_baseline_feature('BoVW', verbose=True)

        bae = AutoEncoderBimodal(X_train_1, X_train_2, X_dev_1, X_dev_2)
        bae.build_model()
        bae.train_model()
    

if __name__ == "__main__":
    unittest.main()