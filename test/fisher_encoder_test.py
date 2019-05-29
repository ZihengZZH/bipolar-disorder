import unittest
import numpy as np
from src.model.fisher_encoder import FisherVectorGMM


class TestFisherVectors(unittest.TestCase):
    def test_fisher_gmm(self):
        np.random.seed(22)
        shape = [200, 30]
        n_kernels = 64
        n_test_videos = 60
        test_data = np.concatenate([np.random.normal(np.zeros(30), size=shape), np.random.normal(np.ones(30), size=shape)], axis=0)

        fv_gmm = FisherVectorGMM(n_kernels=n_kernels)
        fv_gmm.fit(test_data, verbose=1)
        
        fv = fv_gmm.predict(test_data[:n_test_videos])
        self.assertEqual(fv.shape, (n_kernels, 2*30))


if __name__ == '__main__':
    unittest.main()