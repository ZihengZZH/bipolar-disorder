from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.model.text2vec import Text2Vec
from src.model.random_forest import RandomForest
# from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_bags_of_words
from src.metric.uar import get_UAR

import numpy as np
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


def TEXT():
    print("\nrunning doc2vec embeddings on text modality")
    text2vec = Text2Vec(build_on_corpus=True)
    text2vec.load_model()
    feature_name = text2vec.model_name[8:12]
    X_train, y_train = text2vec.load_embedding('train')
    X_dev, y_dev = text2vec.load_embedding('dev')
    rf = RandomForest(feature_name, X_train, y_train, X_dev, y_dev, )
    rf.run()
    y_pred_train, y_pred_dev = rf.evaluate()

    y_train = np.reshape(y_train, (len(y_train), ))
    y_dev = np.reshape(y_dev, (len(y_dev), ))

    get_UAR(y_pred_train, y_train, np.array([]), 'RF', feature_name, 'single', train_set=True, test=False)
    get_UAR(y_pred_dev, y_dev, np.array([]), 'RF', feature_name, 'single')
