from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.model.text2vec import Text2Vec
from src.model.random_forest import RandomForest
from src.model.dnn_classifier import MultiTaskDNN
from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_aligned_features
from src.utils.io import load_bags_of_words
from src.utils.io import load_label
from src.metric.uar import get_UAR

import numpy as np
import pandas as pd


def AE_BOAW(arg):
    print("\nrunning SDAE on BoAW representation")
    X_train_A, X_dev_A, X_test_A, y_train_A, inst_train_A, y_dev_A, inst_dev_A = load_bags_of_words('BoAW', verbose=True)

    ae_boaw = AutoEncoder('BoAW', X_train_A.shape[1])
    ae_boaw.build_model()

    if arg == 'train':
        ae_boaw.train_model(pd.concat([X_train_A, X_dev_A]), X_test_A)
    else:
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


def AE_BOVW(arg):
    print("\nrunning SDAE on BoVW representation")
    X_train_V, X_dev_V, X_test_V, y_train_V, inst_train_V, y_dev_V, inst_dev_V = load_bags_of_words('BoVW', verbose=True)

    ae_bovw = AutoEncoder('BoVW', X_train_V.shape[1])
    ae_bovw.build_model()

    if arg == 'train':
        ae_bovw.train_model(pd.concat([X_train_V, X_dev_V]), X_test_V)
    else:
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


def BAE_BOXW(arg):
    print("\nrunning BiModal AE on XBoW representations")
    X_train_A, X_dev_A, X_test_A, y_train_A, inst_train_A, y_dev_A, inst_dev_A = load_bags_of_words('BoAW', verbose=True)
    X_train_V, X_dev_V, X_test_V, y_train_V, inst_train_V, y_dev_V, inst_dev_V = load_bags_of_words('BoVW', verbose=True)

    bae = AutoEncoderBimodal('bimodal_boxw', X_train_A.shape[1], X_train_V.shape[1])
    bae.build_model()

    if arg == 'train':
        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
    else:
        bae.load_model()
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()

        rf = RandomForest('biAE_boxw', encoded_train, y_train_A, encoded_dev, y_dev_A, test=True)
        rf.run()
        y_pred_train, y_pred_dev = rf.evaluate()

        y_train = np.reshape(y_train_A, (len(y_train_A), ))
        y_dev = np.reshape(y_dev_A, (len(y_dev_A), ))
        inst_train = inst_train_A[0]
        inst_dev = inst_dev_A[0]
        
        get_UAR(y_pred_train, y_train, inst_train, 'RF', 'biAE_boxw', 'multiple', train_set=True, test=True)
        get_UAR(y_pred_dev, y_dev, inst_dev, 'RF', 'biAE_boxw', 'multiple', test=True)


def BAE(arg):
    print("\nrunning BiModal AE on aligned Audio / Video features")
    X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, y_train, inst_train, y_dev, inst_dev = load_aligned_features(verbose=True)

    bae = AutoEncoderBimodal('bimodal_aligned', X_train_A.shape[1], X_train_V.shape[1])
    bae.build_model()

    if arg == 'train':
        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
    else:
        bae.load_model()
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()

        rf = RandomForest('biAE_aligned', encoded_train, y_train, encoded_dev, y_dev, test=True)
        rf.run()
        y_pred_train, y_pred_dev = rf.evaluate()

        y_train = np.reshape(y_train, (len(y_train), ))
        y_dev = np.reshape(y_dev, (len(y_dev), ))

        get_UAR(y_pred_train, y_train, inst_train[0], 'RF', 'biAE_aligned', 'multiple', train_set=True, test=True)
        get_UAR(y_pred_dev, y_dev, inst_dev[0], 'RF', 'biAE_aligned', 'multiple', test=True)


def DNN(arg):
    y_train, inst_train, y_dev, inst_dev = load_aligned_features(no_data=True, verbose=True)

    bae = AutoEncoderBimodal('bimodal_aligned', 1000, 1000)
    encoded_train, encoded_dev = bae.load_presentation()

    ymrs_dev, ymrs_train, _, _ = load_label()
    num_classes = 3

    test_dnn = MultiTaskDNN('bimodal_aligned', encoded_train.shape[1], num_classes)
    y_dev_r = test_dnn.prepare_regression_label(ymrs_dev.values[:, 1], inst_dev)
    y_train_r = test_dnn.prepare_regression_label(ymrs_train.values[:, 1], inst_train)

    assert len(y_train) == len(y_train_r)
    assert len(y_dev) == len(y_dev_r)
    
    test_dnn.build_model()
    test_dnn.train_model(encoded_train, y_train, y_train_r, encoded_dev, y_dev, y_dev_r)
    test_dnn.evaluate_model(encoded_train, y_train, y_train_r, encoded_dev, y_dev, y_dev_r)


def TEXT(arg):
    print("\nrunning doc2vec embeddings on text modality")
    text2vec = Text2Vec(build_on_corpus=True)
    if arg == 'train':
        text2vec.train_model()
    else:
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
