import os
import json
import pickle
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

        self.name = 'kernels%d_convars%s_bayes%d' % (n_kernels, convars_type, use_bayesian)
        self.n_kernels = n_kernels
        self.convars_type = convars_type
        self.use_bayesian = use_bayesian
        self.fitted = False
        self.config = json.load(open('./config/model.json', 'r'))['fisher_vector']
        self.save_dir = self.config['save_dir']
        self.data_dir = self.config['data_dir']
        self.means = None
        self.covars = None
        self.weights = None

        if not self.use_bayesian:
            self.gmm = GaussianMixture(
                        n_components=self.n_kernels,
                        covariance_type=self.convars_type,
                        max_iter=1000,
                        verbose=2)
        else:
            self.gmm = BayesianGaussianMixture(
                        n_components=self.n_kernels,
                        covariance_type=self.convars_type,
                        max_iter=1000,
                        verbose=2)

    def fit(self, X):
        # para X: shape [n_frames, n_features, n_feature_dim]
        # if os.path.isfile(os.path.join(self.save_dir, self.name, 'gmm.model')):
        #     print("\nmodel already trained ---", self.name)
        #     self.load()
        #     return 
        # elif not os.path.isdir(os.path.join(self.save_dir, self.name)):
        #     os.mkdir(os.path.join(self.save_dir, self.name))
        
        self.feature_dim = X.shape[-1]
        # X = X.reshape(-1, X.shape[-1])
        print("\nfitting data into GMM with %d kernels" % self.n_kernels)
        
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
        print("\nmodel trained ---", self.name)
        # self.save()
    
    def score(self, X):
        return self.gmm.score(X.reshape(-1, X.shape[-1]))

    def predict(self, X, normalized=True):
        # para X: shape [n_frames, n_feature_dim]
        assert X.ndim == 2
        assert X.shape[0] >= self.n_kernels, 'n_frames should be greater than n_kernels'

        print("\ninferring fisher vectors with given GMM ...")

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

        # fisher_vector[fisher_vector < 10**-4] = 0 # threshold
        print("\ninferring completed.")
        
        assert fisher_vector.ndim == 2
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
        with open(os.path.join(self.save_dir, self.name, 'gmm.model'), 'wb') as out_gmm:
            pickle.dump(self.gmm, out_gmm, protocol=3)
        with open(os.path.join(self.save_dir, self.name, 'covars.data'), 'wb') as out_covars:
            pickle.dump(self.covars, out_covars, protocol=3)
        out_gmm.close()
        out_covars.close()
        print("\nmodel saved. --- ", self.name)

    def load(self):
        with open(os.path.join(self.save_dir, self.name, 'gmm.model'), 'rb') as in_gmm:
            self.gmm = pickle.load(in_gmm)
        with open(os.path.join(self.save_dir, self.name, 'covars.data'), 'rb') as in_covars:
            self.covars = pickle.load(in_covars)
        in_gmm.close()
        in_covars.close()
        if not self.use_bayesian:
            assert isinstance(self.gmm, GaussianMixture)
        else:
            assert isinstance(self.gmm, BayesianGaussianMixture)
        self.means = self.gmm.means_
        self.weights = self.gmm.weights_
        print("\nmodel loaded. --- ", self.name)

    def save_vector(self, fisher_vector, partition, dynamics=False, label=False):
        if not label:
            filename = 'vector_%s' % partition if dynamics else 'fisher_vector_%s' % partition
            np.save(os.path.join(self.data_dir, filename), fisher_vector)
        else:
            filename = 'label_%s' % partition
            np.save(os.path.join(self.data_dir, filename), fisher_vector)

    def load_vector(self, partition, dynamics=False, label=False):
        if not label:
            filename = 'vector_%s.npy' % partition if dynamics else 'fisher_vector_%s.npy' % partition
            fisher_vector = np.load(os.path.join(self.data_dir, filename), allow_pickle=True)
            return fisher_vector
        else:
            filename = 'label_%s.npy' % partition
            label = np.load(os.path.join(self.data_dir, filename))
            return label


class FisherVectorGMM_BIC():
    def __init__(self):
        self.config = json.load(open('./config/model.json', 'r'))['fisher_vector']
    
    def prepare_data(self, data_train, data_dev):
        fv_gmm = FisherVectorGMM()
        fv_gmm.save_vector(data_train, 'train', dynamics=True)
        fv_gmm.save_vector(data_dev, 'dev', dynamics=True)
    
    def train_model(self):
        kernels = self.config['kernels']
        bic_scores = []
        output = open(os.path.join(self.config['save_dir'], 'best_kernel.txt'), 'w+')
        for kernel in kernels:
            fv_gmm = FisherVectorGMM(n_kernels=kernel)
            X_train = fv_gmm.load_vector('train', dynamics=True)
            X_dev = fv_gmm.load_vector('dev', dynamics=True)
            X = np.vstack((np.vstack(X_train), np.vstack(X_dev)))
            fv_gmm.fit(X)

            bic_score = fv_gmm.score(X)
            bic_scores.append(bic_score)

            print("\nBIC score for kernels %d is --- %.4f" % (kernel, bic_score))
            output.write("\nBIC score for kernels %d is --- %.4f\n" % (kernel, bic_score))
        
        best_bic_score = min(bic_scores)
        best_kernel = kernels[np.argmin(bic_scores)]
        print("\nselected GMM with %d kernels" % best_kernel)
        output.write("\nbest kernel is %d\nbest BIC score is %.4f" % (best_kernel, best_bic_score))
        
        output.close()
        