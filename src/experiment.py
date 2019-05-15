from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_bags_of_words


def AE_BOW():
    print("\nrunning SDAE on BoAW representation")
    X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('BoAW', verbose=True)

    ae_bow = AutoEncoder('BoAW', X_train, X_dev, noisy=False)
    ae_bow.build_model()
    ae_bow.train_model()


def BAE_XBOW():
    print("\nrunning BiModal AE on XBoW representations")
    X_train_A, X_dev_A, X_test_A = load_bags_of_words('BoAW', verbose=True)
    X_train_V, X_dev_V, X_test_V = load_bags_of_words('BoVW', verbose=True)

    import pandas as pd
    X_train_A = pd.concat([X_train_A, X_dev_A])
    X_train_V = pd.concat([X_train_V, X_dev_V])

    assert X_train_A.shape == X_train_V.shape
    assert X_test_A.shape == X_test_V.shape

    bae = AutoEncoderBimodal(X_train_A, X_train_V, X_test_A, X_test_V)
    bae.build_model()
    bae.train_model()
