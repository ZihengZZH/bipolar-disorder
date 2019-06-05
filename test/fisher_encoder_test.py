import unittest
import numpy as np
from src.model.fisher_encoder import FisherVectorGMM
from src.model.fisher_encoder import FisherVectorGMM_BIC


class TestFisherVectors(unittest.TestCase):
    def test_fisher_gmm(self):
        np.random.seed(22)
        shape = [200, 30]
        n_kernels = 20
        n_test_videos = 60
        test_data = np.concatenate([np.random.normal(np.zeros(30), size=shape), np.random.normal(np.ones(30), size=shape)], axis=0)

        # fv_gmm = FisherVectorGMM(n_kernels=n_kernels)
        fv_gmm_bic = FisherVectorGMM_BIC()
        # fv_gmm.fit(test_data)
        fv_gmm_bic.prepare_data(test_data, test_data[:n_test_videos])
        fv_gmm_bic.train_model()
        
        # fv = fv_gmm.predict(test_data[:n_test_videos])
        # self.assertEqual(fv.shape, (n_kernels, 2*30))

    def test_fisher_vector(self):
        fv_gmm = FisherVectorGMM(n_kernels=64)
        X_train_session = fv_gmm.load_vector('train', dynamics=True)
        X_dev_session = fv_gmm.load_vector('dev', dynamics=True)
        print(X_train_session.shape, X_dev_session.shape)
        print(X_train_session[0].shape, X_dev_session[0].shape)
        # assert X_train_session.ndim == X_dev_session.ndim == 3
        print(np.vstack(X_train_session).shape)
        print(np.vstack(X_dev_session).shape)
        fv_gmm.fit(np.vstack((np.vstack(X_train_session), np.vstack(X_dev_session))))

if __name__ == '__main__':
    unittest.main()