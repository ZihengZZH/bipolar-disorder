import sys, getopt


def display_help():
    print("--" * 20)
    print("Several options to run")
    print("--" * 20)
    print("baseline system")
    print("\tmain.py -b")
    print("experiment system")
    print("\tmain.py -x")
    print("landmark visualization")
    print("\tmain.py -v <partition>")
    print("help list")
    print("\tmain.py -h")
    print("--" * 20)


def run_baseline_system():
    from src.baseline import BaseLine
    models = ['SVM', 'RF']
    features = ['MFCC', 'eGeMAPS', 'Deep', 'BoAW', 'AU', 'BoVW']
    print("--" * 20)
    print("Available models:")
    for idx, m in enumerate(models):
        print(idx, m)
    model_id = int(input("choose a model: "))
    print("Available features:")
    for idx, f in enumerate(features):
        print(idx, f)
    feature_id = int(input("choose a feature: "))
    baseline = BaseLine(models[model_id], features[feature_id])
    baseline.run()


# def main(argv):
#     try:
#         opts, _ = getopt.getopt(argv, "hbv:x:", ["help", "baseline", "visualize", "experiment"])
#     except getopt.GetoptError as err:
#         print(err)
#         display_help()
#         sys.exit(2)
    
#     for opt, arg in opts:
#         if opt in ('-h', '--help'):
#             display_help()
#         elif opt in ('-b', '--baseline'):
#             print("Baseline System")
#             print("--" * 20)
#             run_baseline_system()
#         elif opt in ('-x', '--experiment'):
#             from src.experiment import BAE, BAE_BOXW, TEXT, DNN
#             print("Experiment System")
#             print("--" * 20)
#             DNN(arg)
#         elif opt in ('-v', '--visualize'):
#             from src.utils.vis import visualize_landmarks
#             print("Visualize facial landmarks on videos")
#             print("--" * 20)
#             visualize_landmarks(arg)


# if __name__ == "__main__":
    # main(sys.argv[1:])

import numpy as np
import pdb

from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.

    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors

    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.

    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.

    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf

    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def main():
    # Short demo.
    K = 64
    N = 1000

    xx, _ = make_classification(n_samples=N)
    xx_tr, xx_te = xx[: -100], xx[-100: ]

    gmm = GaussianMixture(n_components=K, covariance_type='diag')
    gmm.fit(xx_tr)

    fv = fisher_vector(xx_te, gmm)
    pdb.set_trace()


if __name__ == '__main__':
    main()