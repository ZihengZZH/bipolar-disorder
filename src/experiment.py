from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.model.autoencoder_multimodal import AutoEncoderMultimodal
from src.model.fisher_encoder import FisherVectorGMM
from src.model.fisher_encoder import FisherVectorGMM_BIC
from src.model.text2vec import Text2Vec
from src.model.random_forest import RandomForest
from src.model.dnn_classifier import SingleTaskDNN
from src.model.dnn_classifier import MultiTaskDNN
from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_aligned_features
from src.utils.io import load_bags_of_words
from src.utils.io import load_label
from src.utils.preprocess import upsample
from src.utils.preprocess import get_dynamics
from src.utils.preprocess import frame2session
from src.metric.uar import get_UAR

import numpy as np
import pandas as pd


class Experiment():
    def __init__(self):
        self.display_help()

    def display_help(self):
        self.func_list = ['proposed architecture',
                    'SDAE on unimodality', 'BiSDAE on aligned A/V',
                    'BiSDAE on aligned A/V',
                    'FV using GMM on latent repres', 
                    'DNN as classifier', 'doc2vec on text']
        print("--" * 20)
        print("Experiment")
        print("--" * 20)
        for idx, name in enumerate(self.func_list):
            print(idx, name)
        print("--" * 20)
    
    def run(self):
        choice = int(input("choose a function "))
        if choice > len(self.func_list):
            return 
        if choice == 0:
            self.main_system()
        elif choice == 1:
            self.AE_single()
        elif choice == 2:
            self.BAE_bimodal()
        elif choice == 3:
            self.BAEV_multimodal()
        elif choice == 4:
            self.FV_GMM()
        elif choice == 5:
            self.RF()
        elif choice == 6:
            self.TEXT()
    
    def main_system(self):
        print("\nrunning the proposed architecture (BiSDAE + FV + DNN)")

    def BAE_BOXW(self):
        print("\nrunning BiModal SDAE on XBoW representations")
        X_train_A, X_dev_A, X_test_A, y_train_A, inst_train_A, y_dev_A, inst_dev_A = load_bags_of_words('BoAW', verbose=True)
        X_train_V, X_dev_V, X_test_V, y_train_V, inst_train_V, y_dev_V, inst_dev_V = load_bags_of_words('BoVW', verbose=True)

        bae = AutoEncoderBimodal('bimodal_boxw', X_train_A.shape[1], X_train_V.shape[1])
        bae.build_model()

        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
        
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()

    def AE_single(self):
        print("\nrunning SDAE on unimodal features (facial landmarks and MFCC/eGeMAPS)")
        print("\nchoose a modality\n0.facial landmarks\n1.MFCC\n2.eGeMAPS")
        choice = int(input("choose a function "))
        if choice == 0:
            _, _, _, X_train_V, X_dev_V, X_test_V, _, _, _, _ = load_aligned_features(verbose=True)
            ae = AutoEncoder('unimodal_landmark', 136)
            ae.build_model()
            ae.train_model(pd.concat([X_train_V, X_dev_V]), X_test_V)
            ae.encode(X_train_V, X_dev_V)
        elif choice == 1:
            X_train_A, X_dev_A, X_test_A, _, _, _, _, _, _, _ = load_aligned_features(verbose=True)
            ae = AutoEncoder('unimodal_mfcc', X_train_A.shape[1], visual=False)
            ae.build_model()
            ae.train_model(pd.concat([X_train_A, X_dev_A]), X_test_A)
            ae.encode(X_train_A, X_dev_A)
        elif choice == 2:
            X_train_A, X_dev_A, X_test_A, _, _, _, _, _, _, _ = load_aligned_features(eGeMAPS=True, verbose=True)
            ae = AutoEncoder('unimodal_egemaps', X_train_A.shape[1])
            ae.build_model()
            ae.train_model(pd.concat([X_train_A, X_dev_A]), X_test_A)
            ae.encode(X_train_A, X_dev_A)
    
    def BAE_bimodal(self):
        print("\nrunning BiModal SDAE on aligned Audio / Video features")
        X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, _, _, _, _ = load_aligned_features(verbose=True)

        bae = AutoEncoderBimodal('bimodal_aligned_mfcc', X_train_A.shape[1], 136, noisy=False)
        bae.build_model()

        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
        
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()

    def BAEV_multimodal(self):
        print("\nrunning BiModal SDAE on aligned Audio / Video features (gaze / pose / AUs)")
        X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, y_train, inst_train, y_dev, inst_dev = load_aligned_features(verbose=True)

        mae = AutoEncoderMultimodal('multimodal_aligned_mfcc', X_train_A.shape[1], 136, 6, 6, 35, noisy=False)
        mae.build_model()

        mae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
        
        mae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = mae.load_presentation()

    def FV_GMM(self):
        print("\nrunning Fisher Encoder using GMM on learnt representations")
        y_train_frame, inst_train, y_dev_frame, inst_dev = load_aligned_features(no_data=True, verbose=True)
        bae = AutoEncoderBimodal('bimodal_aligned', 117, 136)
        encoded_train, encoded_dev = bae.load_presentation()
        assert len(y_train_frame) == len(inst_train) == len(encoded_train)
        assert len(y_dev_frame) == len(inst_dev) == len(encoded_dev)

        fv_BIC = FisherVectorGMM_BIC()
        
        X_train, y_train = frame2session(encoded_train, y_train_frame, inst_train, verbose=True)
        X_dev, y_dev = frame2session(encoded_dev, y_dev_frame, inst_dev, verbose=True)
        print(y_train.shape, y_dev.shape)

        X_train_session = [get_dynamics(train) for train in X_train]
        X_dev_session = [get_dynamics(dev) for dev in X_dev]

        fv_BIC.prepare_data(X_train_session, X_dev_session)
        fv_BIC.train_model()

        if False:
            fv_gmm = FisherVectorGMM(n_kernels=32)
            fv_gmm.load()

            X_train = fv_gmm.load_vector('train', dynamics=True)
            X_dev = fv_gmm.load_vector('dev', dynamics=True)
            
            # after n_kernels is determined
            fv_train = np.array([fv_gmm.predict(train) for train in X_train])
            fv_dev = np.array([fv_gmm.predict(dev) for dev in X_dev])

            fv_gmm.save_vector(fv_train, 'train', dynamics=False)
            fv_gmm.save_vector(fv_dev, 'dev', dynamics=False)


    def DNN(self):
        fv_gmm = FisherVectorGMM(n_kernels=32)
        fv_gmm.load()

        X_train = fv_gmm.load_vector('train', dynamics=False)
        X_dev = fv_gmm.load_vector('dev', dynamics=False)
        print(X_train.shape, X_dev.shape)
        X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
        X_dev = X_dev.reshape((X_dev.shape[0], np.prod(X_dev.shape[1:])))
        print(X_train.shape, X_dev.shape)

        y_dev_r, y_train_r, y_dev, y_train = load_label()
        y_train = y_train.astype('int')
        y_dev = y_dev.astype('int')
        num_classes = 3

        test_dnn = MultiTaskDNN('fv_gmm', X_train.shape[1], num_classes)
        
        assert len(y_train) == len(y_train_r) == X_train.shape[0]
        assert len(y_dev) == len(y_dev_r) == X_dev.shape[0]
        
        test_dnn.build_model()
        test_dnn.train_model(X_train, y_train, y_train_r, X_dev, y_dev, y_dev_r)
        test_dnn.evaluate_model(X_train, y_train, y_train_r, X_dev, y_dev, y_dev_r)


    def TEXT(self):
        print("\nrunning doc2vec embeddings on text modality")
        text2vec = Text2Vec(build_on_corpus=True)
        text2vec.build_model()
        text2vec.train_model()
        text2vec.infer_embedding('train')
        text2vec.infer_embedding('dev')


    def RF(self):
        fv_gmm = FisherVectorGMM(n_kernels=32)
        fv_gmm.load()

        X_train = fv_gmm.load_vector('train', dynamics=False)
        X_dev = fv_gmm.load_vector('dev', dynamics=False)
        print(X_train.shape, X_dev.shape)
        X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
        X_dev = X_dev.reshape((X_dev.shape[0], np.prod(X_dev.shape[1:])))
        print(X_train.shape, X_dev.shape)

        y_dev_r, y_train_r, y_dev, y_train = load_label()
        y_train = y_train.astype('int')
        y_dev = y_dev.astype('int')

        rf = RandomForest('fv_gmm', X_train, y_train, X_dev, y_dev, test=True)
        rf.run()
        y_pred_train, y_pred_dev = rf.evaluate()
        
        get_UAR(y_pred_train, y_train, np.array([]), 'fv_gmm', 'fv_gmm', 'multi', train_set=True, test=True)
        get_UAR(y_pred_dev, y_dev, np.array([]), 'fv_gmm', 'fv_gmm', 'multi', test=True)