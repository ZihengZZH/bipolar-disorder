import os
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture


class FisherVectorGMM:
    """
    Fisher Vector derived from GMM
    ---
    Attributes
    -----------
    n_kernels: int
        number of kernels in GMM
    convars_type: str
        convariance type for GMM
    use_bayesian: bool
        whether or not to use Baysian GMM
    gmm: GaussianMixture() or BayesianGaussianMixture()
        GMM instance in sklearn
    means: np.array()
        means learned in GMM
    covars: np.array()
        covariance learned in GMM
    weights: np.array()
        weights learned in GMM
    ---------------------------------------
    Functions
    -----------
    fit(): public
        fit raw data into GMM
    predict(): public
        predict FV for one video (variable frames)
    predict_alternative(): public
        predict FV for one video (variable frames) alternative
        not validated
    save(): public
        save GMM model into external file
    load(): public
        load GMM model from external file
    """
    def __init__(self, n_kernels=1, convars_type='diag', use_bayesian=False):
        # para n_kernels:
        # para convars_type:
        # para use_bayesian:
        assert convars_type in ['diag', 'full']
        assert n_kernels > 0

        self.n_kernels = n_kernels
        self.convars_type = convars_type
        self.use_bayesian = use_bayesian
        self.gmm = None
        self.means = None
        self.covars = None
        self.weights = None
        self.feature_dim = 0
        self.config = json.load(open('./config/model.json', 'r'))['fisher_vector']
        self.save_dir = self.config['save_dir']
        self.save_dir_vec = self.config['save_dir_vector']

    def fit(self, X, verbose=0):
        # para X: shape [n_videos, n_frames, n_features, n_feature_dim]
        self.feature_dim = X.shape[-1]
        X = X.reshape(-1, X.shape[-1])
        print("\nfitting data into GMM with %d kernels" % self.n_kernels)

        if not self.use_bayesian:
            self.gmm = GaussianMixture(
                        n_components=self.n_kernels,
                        covariance_type=self.convars_type,
                        max_iter=1000,
                        verbose=verbose)
        else:
            self.gmm = BayesianGaussianMixture(
                        n_components=self.n_kernels,
                        covariance_type=self.convars_type,
                        max_iter=1000,
                        verbose=verbose)
        
        self.gmm.fit(X)
        self.means = self.gmm.means_
        self.covars = self.gmm.covariances_
        self.weights = self.gmm.weights_
        print("\nfitting completed")

        # if cov_type is diagonal - make sure that covars holds a diagonal matrix
        if self.convars_type == 'diag':
            cov_matrices = np.empty(shape=(self.n_kernels, self.covars.shape[1], self.covars.shape[1]))
            for i in range(self.n_kernels):
                cov_matrices[i, :, :] = np.diag(self.covars[i, :])
            self.covars = cov_matrices

        assert self.covars.ndim == 3
        self.save()

    def predict(self, X, normalized=True, partition='train'):
        # para X: shape [n_frames, n_feature_dim]
        assert X.ndim == 2
        assert self.feature_dim == X.shape[-1]

        X_matrix = X.reshape(-1, X.shape[-1]) # [n_frames, n_feature_dim]
        
        # set equal weights to predict likelihood ratio
        self.gmm.weights_ = np.ones(self.n_kernels) / self.n_kernels
        likelihood_ratio = self.gmm.predict_proba(X_matrix).reshape(X.shape[0], self.n_kernels) # [n_frames, n_kernels]

        var = np.diagonal(self.covars, axis1=1, axis2=2) # [n_kernels, n_feature_dim]

        # decrease the memory use
        norm_dev_from_modes = np.tile(X[:,None,:],(1,self.n_kernels,1)) 
        np.subtract(norm_dev_from_modes, self.means[None, :], out=norm_dev_from_modes)
        np.divide(norm_dev_from_modes, var[None, :], out=norm_dev_from_modes)
        """
        norm_dev_from_modes:
            (X - mean) / var
            [n_frames, n_kernels, n_feature_dim]
        """

        # mean deviation
        mean_dev = np.multiply(likelihood_ratio[:,:,None], norm_dev_from_modes).mean(axis=0) # [n_kernels, n_feature_dim]
        mean_dev = np.multiply(1 / np.sqrt(self.weights[:,None]), mean_dev) # [n_kernels, n_feature_dim]

        # covariance deviation
        cov_dev = np.multiply(likelihood_ratio[:,:, None], norm_dev_from_modes**2 - 1).mean(axis=0) # [n_kernels, n_feature_dim]
        cov_dev = np.multiply(1 / np.sqrt(2 * self.weights[:,  None]), cov_dev) # [n_kernels, n_feature_dim]

        # stack vectors of mean and covariance
        fisher_vector = np.concatenate([mean_dev, cov_dev], axis=1)

        if normalized:
            fisher_vector = np.sqrt(np.abs(fisher_vector)) * np.sign(fisher_vector) # power normalization
            fisher_vector = fisher_vector / np.linalg.norm(fisher_vector, axis=0) # L2 normalization

        fisher_vector[fisher_vector < 10**-4] = 0 # threshold

        assert fisher_vector.ndim == 2
        self.save_vector(fisher_vector, partition)
        return fisher_vector

    def predict_alternative(self, X, normalized=True):
        X = np.atleast_2d(X)
        N = X.shape[0]

        # Compute posterior probabilities.
        Q = self.gmm.predict_proba(X)  # NxK

        # Compute the sufficient statistics of descriptors.
        Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
        Q_X = np.dot(Q.T, X) / N
        Q_XX_2 = np.dot(Q.T, X ** 2) / N

        # compute derivatives with respect to mixing weights, means and variances.
        d_pi = Q_sum.squeeze() - self.gmm.weights_
        d_mu = Q_X - Q_sum * self.gmm.means_
        d_sigma = (
            - Q_XX_2
            - Q_sum * self.gmm.means_ ** 2
            + Q_sum * self.gmm.covariances_
            + 2 * Q_X * self.gmm.means_)

        # merge derivatives into a vector.
        fisher_vector = np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

        if normalized:
            fisher_vector = np.sqrt(np.abs(fisher_vector)) * np.sign(fisher_vector) # power normalization
            fisher_vector = fisher_vector / np.linalg.norm(fisher_vector, axis=0) # L2 norm

        return fisher_vector

    def save(self):
        filename = 'kernel%d_%s_bayes%d.npz' % (self.n_kernels, self.convars_type, self.use_bayesian)
        np.savez(os.path.join(self.save_dir, filename), means=self.means, covars=self.covars, weights=self.weights)

    def load(self):
        filename = '%d_%s_bayes%d.npz' % (self.n_kernels, self.convars_type, self.use_bayesian)
        npzfile = np.load(os.path.join(self.save_dir, filename))

        self.means = npzfile['means']
        self.covars = npzfile['covars']
        self.weights = npzfile['weights']
        
        self.gmm.weights_ = self.weights

    def save_vector(self, fisher_vector, partition):
        np.save(os.path.join(self.save_dir_vec, 'vector_%s' % partition), fisher_vector)

    def load_vector(self, partition):
        fisher_vector = np.load(os.path.join(self.save_dir_vec, 'vector_%s.npy' % partition))
        return fisher_vector
