from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.model.autoencoder_bimodal import AutoEncoderBimodalV
from src.model.fisher_encoder import FisherVectorGMM
from src.model.text2vec import Text2Vec
from src.model.random_forest import RandomForest
from src.model.dnn_classifier import MultiTaskDNN
from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_aligned_features
from src.utils.io import load_bags_of_words
from src.utils.io import load_label
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
                    'BiSDAE on BoXW', 'BiSDAE on aligned A/V',
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
            self.BAE_BOXW()
        elif choice == 2:
            self.BAE()
        elif choice == 3:
            self.FV_GMM()
        elif choice == 4:
            self.DNN()
        elif choice == 5:
            self.TEXT()
    
    def main_system(self):
        print("\nrunning the proposed architecture (BiSDAE + FV + DNN)")
        # load features [FRAME-LEVEL]
        X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, y_train_frame, inst_train, y_dev_frame, inst_dev = load_aligned_features(verbose=True)
        # BiSDAE model to learn audio-visual features
        bae = AutoEncoderBimodal('proposed_arch', X_train_A.shape[1], X_train_V.shape[1])
        bae.build_model()

        # train BiSDAE model
        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
        
        # encode with pre-trained model
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()
        assert len(y_train_frame) == len(inst_train) == len(encoded_train)
        assert len(y_dev_frame) == len(inst_dev) == len(encoded_dev)

        # FV GMM model to captures temporary information
        fv_gmm = FisherVectorGMM(n_kernels=64)
        X_train_frame, X_dev_frame = get_dynamics(encoded_train), get_dynamics(encoded_dev)
        # save data along with dynamics 
        fv_gmm.save_vector(X_train_frame, 'train', dynamics=True)
        fv_gmm.save_vector(X_dev_frame, 'dev', dynamics=True)
        
        # fit GMM with 64 kernels
        fv_gmm.fit(np.vstack((X_train_frame, X_dev_frame)), verbose=2)

        X_train, y_train = frame2session(X_train_frame, y_train_frame, inst_train, verbose=True)
        X_dev, y_dev = frame2session(X_dev_frame, y_dev_frame, inst_dev, verbose=True)
        # produce FV for train/dev data
        fv_train = fv_gmm.predict(X_train, partition='train')
        fv_dev = fv_gmm.predict(X_dev, partition='dev')
        # prepare regression data
        ymrs_dev, ymrs_train, _, _ = load_label()
        num_classes = 3

        # multi-task DNN to classify learned features
        multi_dnn = MultiTaskDNN('proposed_arch', fv_train.shape[1], num_classes)
        multi_dnn.build_model()
        # prepare labels for regression task
        y_dev_r = multi_dnn.prepare_regression_label(ymrs_dev.values[:, 1], inst_dev)
        y_train_r = multi_dnn.prepare_regression_label(ymrs_train.values[:, 1], inst_train)
        assert len(y_train) == len(y_train_r)
        assert len(y_dev) == len(y_dev_r)

        multi_dnn.train_model(encoded_train, y_train, y_train_r, encoded_dev, y_dev, y_dev_r)
        multi_dnn.evaluate_model(encoded_train, y_train, y_train_r, encoded_dev, y_dev, y_dev_r)

    def BAE_BOXW(self):
        print("\nrunning BiModal AE on XBoW representations")
        X_train_A, X_dev_A, X_test_A, y_train_A, inst_train_A, y_dev_A, inst_dev_A = load_bags_of_words('BoAW', verbose=True)
        X_train_V, X_dev_V, X_test_V, y_train_V, inst_train_V, y_dev_V, inst_dev_V = load_bags_of_words('BoVW', verbose=True)

        bae = AutoEncoderBimodal('bimodal_boxw', X_train_A.shape[1], X_train_V.shape[1])
        bae.build_model()

        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
        
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()

    def BAE(self):
        print("\nrunning BiModal AE on aligned Audio / Video features")
        X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, y_train, inst_train, y_dev, inst_dev = load_aligned_features(verbose=True)

        bae = AutoEncoderBimodalV('bimodalV_aligned', 117, 136, 6, 6, 35)
        bae.build_model()

        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
        
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()

    def FV_GMM(self):
        print("\nrunning Fisher Encoder using GMM on learnt representations")
        y_train_frame, inst_train, y_dev_frame, inst_dev = load_aligned_features(no_data=True, verbose=True)
        bae = AutoEncoderBimodal('bimodal_aligned', 118, 184)
        encoded_train, encoded_dev = bae.load_presentation()
        assert len(y_train_frame) == len(inst_train) == len(encoded_train)
        assert len(y_dev_frame) == len(inst_dev) == len(encoded_dev)

        fv_gmm = FisherVectorGMM(n_kernels=64)
        X_train_frame, X_dev_frame = get_dynamics(encoded_train), get_dynamics(encoded_dev)
        fv_gmm.save_vector(X_train_frame, 'train', dynamics=True)
        fv_gmm.save_vector(X_dev_frame, 'dev', dynamics=True)
        X_train_frame = fv_gmm.load_vector('train', dynamics=True)
        X_dev_frame = fv_gmm.load_vector('dev', dynamics=True)
        fv_gmm.fit(np.vstack((X_train_frame, X_dev_frame)), verbose=2)
        X_train, y_train = frame2session(X_train_frame, y_train_frame, inst_train, verbose=True)
        X_dev, y_dev = frame2session(X_dev_frame, y_dev_frame, inst_dev, verbose=True)
        fv_train = fv_gmm.predict(X_train, partition='train')
        fv_dev = fv_gmm.predict(X_dev, partition='dev')

    def DNN(self):
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

    def TEXT(self):
        print("\nrunning doc2vec embeddings on text modality")
        text2vec = Text2Vec(build_on_corpus=True)
        text2vec.build_model()
        text2vec.train_model()
        text2vec.infer_embedding('train')
        text2vec.infer_embedding('dev')
