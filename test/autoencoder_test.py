import numpy as np
import unittest
from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_lstm import AutoEncoderLSTM
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.utils.io import load_proc_baseline_feature


class TestAutoEncoder(unittest.TestCase):
    def test_build_model(self):
        pass
    

if __name__ == "__main__":
    unittest.main()