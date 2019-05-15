from src.model.autoencoder import AutoEncoder
from src.utils.io import load_proc_baseline_feature


def AE_BOW(verbose=False):
    print("\nrunning SDAE on BoAW representation")
    X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('BoAW', verbose=True)

    ae_bow = AutoEncoder('BoAW', X_train, X_dev, noisy=False)
    ae_bow.build_model()
    ae_bow.train_model()
