from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.model.text2vec import Text2Vec
from src.model.random_forest import RandomForest
from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_aligned_features
from src.utils.io import load_bags_of_words
from src.metric.uar import get_UAR

import numpy as np
import pandas as pd


def AE_BOAW():
    print("\nrunning SDAE on BoAW representation")
    X_train_A, X_dev_A, X_test_A, y_train_A, inst_train_A, y_dev_A, inst_dev_A = load_bags_of_words('BoAW', verbose=True)

    ae_boaw = AutoEncoder('BoAW', 
                        pd.concat([X_train_A, X_dev_A]),
                        X_test_A, 
                        noisy=True)
    ae_boaw.build_model()
    # ae_boaw.train_model()
    ae_boaw.load_model()
    ae_boaw.encode(X_train_A, X_dev_A)
    encoded_train, encoded_dev = ae_boaw.load_presentation()

    rf = RandomForest('biAE', encoded_train, y_train_A, encoded_dev, y_dev_A, test=True)
    rf.run()
    y_pred_train, y_pred_dev = rf.evaluate()

    y_train = np.reshape(y_train_A, (len(y_train_A), ))
    y_dev = np.reshape(y_dev_A, (len(y_dev_A), ))

    get_UAR(y_pred_train, y_train, inst_train_A, 'RF', 'BoAW', 'single', train_set=True, test=True)
    get_UAR(y_pred_dev, y_dev, inst_dev_A, 'RF', 'BoVW', 'single', test=True)


def AE_BOVW():
    print("\nrunning SDAE on BoVW representation")
    X_train_V, X_dev_V, X_test_V, y_train_V, inst_train_V, y_dev_V, inst_dev_V = load_bags_of_words('BoVW', verbose=True)

    ae_bovw = AutoEncoder('BoVW', 
                        pd.concat([X_train_V, X_dev_V]),
                        X_test_V, 
                        noisy=True)
    ae_bovw.build_model()
    # ae_bovw.train_model()
    ae_bovw.load_model()
    ae_bovw.encode(X_train_V, X_dev_V)
    encoded_train, encoded_dev = ae_bovw.load_presentation()

    rf = RandomForest('biAE', encoded_train, y_train_V, encoded_dev, y_dev_V, test=True)
    rf.run()
    y_pred_train, y_pred_dev = rf.evaluate()

    y_train = np.reshape(y_train_V, (len(y_train_V), ))
    y_dev = np.reshape(y_dev_V, (len(y_dev_V), ))

    get_UAR(y_pred_train, y_train, inst_train_V, 'RF', 'BoVW', 'single', train_set=True, test=True)
    get_UAR(y_pred_dev, y_dev, inst_dev_V, 'RF', 'BoVW', 'single', test=True)


def BAE_BOXW():
    print("\nrunning BiModal AE on XBoW representations")
    X_train_A, X_dev_A, X_test_A, y_train_A, inst_train_A, y_dev_A, inst_dev_A = load_bags_of_words('BoAW', verbose=True)
    X_train_V, X_dev_V, X_test_V, y_train_V, inst_train_V, y_dev_V, inst_dev_V = load_bags_of_words('BoVW', verbose=True)

    bae = AutoEncoderBimodal(
        pd.concat([X_train_A, X_dev_A]), 
        pd.concat([X_train_V, X_dev_V]), 
        X_test_A, X_test_V)
    
    bae.build_model()
    bae.train_model()
    # bae.load_model()
    # bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
    # encoded_train, encoded_dev = bae.load_presentation()

    # rf = RandomForest('biAE', encoded_train, y_train_A, encoded_dev, y_dev_A, test=True)
    # rf.run()
    # y_pred_train, y_pred_dev = rf.evaluate()

    # y_train = np.reshape(y_train_A, (len(y_train_A), ))
    # y_dev = np.reshape(y_dev_A, (len(y_dev_A), ))

    # get_UAR(y_pred_train, y_train, inst_train_A, 'RF', 'biAE', 'multiple', train_set=True, test=True)
    # get_UAR(y_pred_dev, y_dev, inst_dev_A, 'RF', 'biAE', 'multiple', test=True)


def BAE():
    print("\nrunning BiModal AE on aligned Audio / Video features")
    X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, y_train, inst_train, y_dev, inst_dev = load_aligned_features(verbose=True)

    bae = AutoEncoderBimodal(
        pd.concat([X_train_A, X_dev_A]), 
        pd.concat([X_train_V, X_dev_V]), 
        X_test_A, X_test_V)
    
    bae.build_model()
    bae.train_model()
    # bae.load_model()
    # bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
    # encoded_train, encoded_dev = bae.load_presentation()

    # rf = RandomForest('biAE', encoded_train, y_train, encoded_dev, y_dev, test=True)
    # rf.run()
    # y_pred_train, y_pred_dev = rf.evaluate()

    # y_train = np.reshape(y_train, (len(y_train), ))
    # y_dev = np.reshape(y_dev, (len(y_dev), ))

    # get_UAR(y_pred_train, y_train, inst_train, 'RF', 'biAE', 'multiple', train_set=True, test=True)
    # get_UAR(y_pred_dev, y_dev, inst_dev, 'RF', 'biAE', 'multiple', test=True)


def TEXT():
    print("\nrunning doc2vec embeddings on text modality")
    text2vec = Text2Vec(build_on_corpus=True)
    text2vec.load_model()
    feature_name = text2vec.model_name[8:12]
    X_train, y_train = text2vec.load_embedding('train')
    X_dev, y_dev = text2vec.load_embedding('dev')
    rf = RandomForest(feature_name, X_train, y_train, X_dev, y_dev)
    rf.run()
    y_pred_train, y_pred_dev = rf.evaluate()

    y_train = np.reshape(y_train, (len(y_train), ))
    y_dev = np.reshape(y_dev, (len(y_dev), ))

    get_UAR(y_pred_train, y_train, np.array([]), 'RF', feature_name, 'single', train_set=True, test=False)
    get_UAR(y_pred_dev, y_dev, np.array([]), 'RF', feature_name, 'single')
