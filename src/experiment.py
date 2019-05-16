from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_bags_of_words

import pandas as pd


def AE_BOW():
    print("\nrunning SDAE on BoAW representation")
    X_train_A, X_dev_A, X_test_A = load_bags_of_words('BoAW', verbose=True)
    X_train_A = pd.concat([X_train_A, X_dev_A])

    ae_boaw = AutoEncoder('BoAW', X_train_A, X_test_A, noisy=True)
    ae_boaw.build_model()
    ae_boaw.train_model()

    print("\nrunning SDAE on BoVW representation")
    X_train_V, X_dev_V, X_test_V = load_bags_of_words('BoVW', verbose=True)
    X_train_V = pd.concat([X_train_V, X_dev_V])

    ae_bovw = AutoEncoder('BoVW', X_train_V, X_test_V, noisy=True)
    ae_bovw.build_model()
    ae_bovw.train_model()


def BAE_XBOW():
    print("\nrunning BiModal AE on XBoW representations")
    X_train_A, X_dev_A, X_test_A = load_bags_of_words('BoAW', verbose=True)
    X_train_V, X_dev_V, X_test_V = load_bags_of_words('BoVW', verbose=True)

    X_train_A = pd.concat([X_train_A, X_dev_A])
    X_train_V = pd.concat([X_train_V, X_dev_V])

    assert X_train_A.shape == X_train_V.shape
    assert X_test_A.shape == X_test_V.shape

    bae = AutoEncoderBimodal(X_train_A, X_train_V, X_test_A, X_test_V)
    bae.build_model()
    bae.train_model()
