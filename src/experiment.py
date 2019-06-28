from src.model.autoencoder import AutoEncoder
from src.model.autoencoder_bimodal import AutoEncoderBimodal
from src.model.autoencoder_multimodal import AutoEncoderMultimodal
from src.model.fisher_encoder import FisherVectorGMM
from src.model.fisher_encoder import FisherVectorGMM_BIC
from src.model.text2vec import Text2Vec
from src.model.random_forest import RandomForest
from src.model.dnn_classifier import SingleTaskDNN
from src.model.dnn_classifier import MultiTaskDNN
from src.model.dnn_classifier import MultiLossDNN
from src.utils.io import load_proc_baseline_feature
from src.utils.io import load_aligned_features
from src.utils.io import load_bags_of_words
from src.utils.io import load_label
from src.utils.preprocess import k_fold_cv
from src.utils.preprocess import upsample
from src.utils.preprocess import get_dynamics
from src.utils.preprocess import frame2session
from src.utils.preprocess import preprocess_metadata_tensorboard
from src.metric.uar import get_UAR
from src.stats import PermutationTest

import os
import json
import pymrmr
import numpy as np
import pandas as pd
from smart_open import smart_open


"""
EXPERIMENT or PROPOSED FRAMEWORK
----------
0. pre-align A/V features
1. uni-DDAE / bi-DDAE / multi-DDAE
2. dynamics (1st & 2nd derivatives)
3. GMM fitting (16/32 kernels)
4. improved Fisher Vector
5. Feature Selection (tree-based)
6. doc2vec training (addition Turkish corpus)
7. document embeddings (PV-DBOW / PV-DM)
8. early fusion 
9. RF / multi-task DNN (as classifiers)
10. Monte Carol permutation test (optional)
"""


class Experiment():
    def __init__(self):
        self.kernel = 32
        self.display_help()

    def display_help(self):
        self.func_list = ['proposed architecture',
                    'DDAE on unimodality', 'BiDDAE on aligned A/V',
                    'MultiDDAE on aligned A/V',
                    'Dynamics on latent repres',
                    'FV using GMM on latent repres', 
                    'feature selection on FVs', 'RF on FVs only',
                    'DNN as classifier (audio-visual-textual)', 
                    'RF as classifier (audio-visual-textual)',
                    'RF as classifier CV (audio-visual-textual)',
                    'doc2vec on text', 'RF on doc2vec',
                    'permutation test']
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
            self.DDAE_single()
        elif choice == 2:
            self.DDAE_bimodal()
        elif choice == 3:
            self.DDAE_multimodal()
        elif choice == 4:
            self.DYNAMICS()
        elif choice == 5:
            self.FV_GMM()
        elif choice == 6:
            self.FV_tree()
        elif choice == 7:
            self.FV_RF()
        elif choice == 8:
            self.DNN()
        elif choice == 9:
            self.RF()
        elif choice == 10:
            self.RF_CV()
        elif choice == 11:
            self.TEXT()
        elif choice == 12:
            self.TEXT_RF()
        elif choice == 13:
            self.PERMUTATION()
    
    def main_system(self):
        print("\nrunning the proposed architecture (DDAE + FV + DNN)")

    def DDAE_BOXW(self):
        print("\nrunning BiModal DDAE on XBoW representations")
        X_train_A, X_dev_A, X_test_A, _, _, _, _ = load_bags_of_words('BoAW', verbose=True)
        X_train_V, X_dev_V, X_test_V, _, _, _, _ = load_bags_of_words('BoVW', verbose=True)

        bae = AutoEncoderBimodal('bimodal_boxw', X_train_A.shape[1], X_train_V.shape[1])
        bae.build_model()

        bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                        pd.concat([X_train_V, X_dev_V]), 
                        X_test_A, X_test_V)
        
        bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
        encoded_train, encoded_dev = bae.load_presentation()

    def DDAE_single(self):
        print("\nrunning DDAE on unimodal features (facial landmarks and MFCC/eGeMAPS)")
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
    
    def DDAE_bimodal(self):
        print("\nrunning BiModal DDAE on aligned Audio / Video features")
        print("\nchoose a modality\n1.landmarks + MFCC\n2.landmarks + eGeMAPS")
        choice = int(input("choose a function "))
        if choice == 1:
            X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, _, _, _, _ = load_aligned_features(verbose=True)
            bae = AutoEncoderBimodal('bimodal_aligned_mfcc', X_train_A.shape[1], 136)
            bae.build_model()
            bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                            pd.concat([X_train_V, X_dev_V]), 
                            X_test_A, X_test_V)
            bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
            encoded_train, encoded_dev = bae.load_presentation()
        elif choice == 2:
            X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, _, _, _, _ = load_aligned_features(eGeMAPS=True, verbose=True)
            bae = AutoEncoderBimodal('bimodal_aligned_egemaps', X_train_A.shape[1], 136)
            bae.build_model()
            bae.train_model(pd.concat([X_train_A, X_dev_A]), 
                            pd.concat([X_train_V, X_dev_V]), 
                            X_test_A, X_test_V)
            bae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
            encoded_train, encoded_dev = bae.load_presentation()

    def DDAE_multimodal(self):
        print("\nrunning BiModal DDAE on aligned Audio / Video features (gaze / pose / AUs)")
        print("\nchoose a modality\n1.landmarks + MFCC\n2.landmarks + eGeMAPS")
        choice = int(input("choose a function "))
        if choice == 1:
            X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, _, _, _, _ = load_aligned_features(verbose=True)
            mae = AutoEncoderMultimodal('multimodal_aligned_mfcc', X_train_A.shape[1], 136, 6, 6, 35)
            mae.build_model()
            mae.train_model(pd.concat([X_train_A, X_dev_A]), 
                            pd.concat([X_train_V, X_dev_V]), 
                            X_test_A, X_test_V)
            mae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
            encoded_train, encoded_dev = mae.load_presentation()
        if choice == 2:
            X_train_A, X_dev_A, X_test_A, X_train_V, X_dev_V, X_test_V, _, _, _, _ = load_aligned_features(eGeMAPS=True, verbose=True)
            mae = AutoEncoderMultimodal('multimodal_aligned_egemaps', X_train_A.shape[1], 136, 6, 6, 35)
            mae.build_model()
            mae.train_model(pd.concat([X_train_A, X_dev_A]), 
                            pd.concat([X_train_V, X_dev_V]), 
                            X_test_A, X_test_V)
            mae.encode(X_train_A, X_train_V, X_dev_A, X_dev_V)
            encoded_train, encoded_dev = mae.load_presentation()

    def DYNAMICS(self):
        print("\nrunning computation of dynamics of latent representation learned by DDAEs")
        ae = AutoEncoder('dynamics', 0)

        with smart_open(os.path.join(ae.save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                print(line_no, '\t', line)
                
                if os.path.isfile(os.path.join(line, 'encoded_train_dynamics.npy')) and os.path.isfile(os.path.join(line, 'encoded_dev_dynamics.npy')):
                    continue
                
                X_train = np.load(os.path.join(line, 'encoded_train.npy'))
                X_dev = np.load(os.path.join(line, 'encoded_dev.npy'))
                X_train_frame = get_dynamics(X_train)
                X_dev_frame = get_dynamics(X_dev)

                assert X_train_frame.shape[0] == X_train.shape[0] - 2
                assert X_dev_frame.shape[0] == X_dev.shape[0] - 2

                print("Shape of training data", X_train.shape)
                print("Shape of development data", X_dev.shape)
                print("Shape of training data with dynamics", X_train_frame.shape)
                print("Shape of development data with dynamics", X_dev_frame.shape)

                np.save(os.path.join(line, 'encoded_train_dynamics'), X_train_frame)
                np.save(os.path.join(line, 'encoded_dev_dynamics'), X_dev_frame)

                print("\ncomputing dynamics done\n")
                del X_train, X_train_frame, X_dev, X_dev_frame
    
    def FV_BIC(self):
        print("\nrunning Fisher Encoder using BIC selections on learnt representations with dynamics")
        fv_gmm_bic = FisherVectorGMM_BIC()

        line = './pre-trained/DDAE/bimodal_aligned_mfcc_hidden0.50_batch1024_epoch100_noise0.1'

        X_train_frame = np.load(os.path.join(line, 'encoded_train_dynamics.npy'))
        X_dev_frame = np.load(os.path.join(line, 'encoded_dev_dynamics.npy'))

        fv_gmm_bic.prepare_data(X_train_frame, X_dev_frame)
        fv_gmm_bic.train_model()

    def FV_GMM(self):
        print("\nrunning Fisher Encoder using GMM on learnt representations with dynamics")

        y_train_frame, inst_train, y_dev_frame, inst_dev = load_aligned_features(no_data=True, verbose=True)
        y_train_frame, y_dev_frame = y_train_frame[2:,:], y_dev_frame[2:,:]
        inst_train, inst_dev = inst_train[2:,:], inst_dev[2:,:]

        ae = AutoEncoder('fv_gmm', 0)

        with smart_open(os.path.join(ae.save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                print(line_no, '\t', line[65:])

                if os.path.isfile(os.path.join(line, 'fisher_vector_train_%d.npy' % self.kernel)) and os.path.isfile(os.path.join(line, 'fisher_vector_dev_%d.npy' % self.kernel)):
                    preprocess_metadata_tensorboard(line, self.kernel)
                    continue
                
                X_train_frame = np.load(os.path.join(line, 'encoded_train_dynamics.npy'))
                X_dev_frame = np.load(os.path.join(line, 'encoded_dev_dynamics.npy'))

                X_train_session, y_train_session = frame2session(X_train_frame, y_train_frame, inst_train, verbose=True)
                X_dev_session, y_dev_session = frame2session(X_dev_frame, y_dev_frame, inst_dev, verbose=True)

                fv_train, fv_dev = [], []
                score = []

                for X_train in X_train_session:
                    fv_gmm = FisherVectorGMM(n_kernels=self.kernel)
                    fv_gmm.fit(X_train)
                    fv_train.append(fv_gmm.predict(X_train, normalized=False))
                    score.append(fv_gmm.score(X_train))
                
                for X_dev in X_dev_session:
                    fv_gmm = FisherVectorGMM(n_kernels=self.kernel)
                    fv_gmm.fit(X_dev)
                    fv_dev.append(fv_gmm.predict(X_dev, normalized=False))
                    score.append(fv_gmm.score(X_dev))

                print("\nscores for all the FV kernel", score)
                fv_gmm.data_dir = line

                fv_gmm.save_vector(fv_train, 'train')
                fv_gmm.save_vector(fv_dev, 'dev')
                fv_gmm.save_vector(y_train_session, 'train', label=True)
                fv_gmm.save_vector(y_dev_session, 'dev', label=True)
                print("\nFV encoding for %s, done" % line[65:])
    
    def FV_mRMR(self):
        print("\nrunning mRMR algorithm for feature selection")
        ae = AutoEncoder('fv_gmm', 0)

        with smart_open(os.path.join(ae.save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                print(line_no, '\t', line[65:])

                if os.path.isfile(os.path.join(line, 'fisher_vector_train_%d.npy' % self.kernel)) and os.path.isfile(os.path.join(line, 'fisher_vector_dev_%d.npy' % self.kernel)):
                    X_train = np.load(os.path.join(line, 'fisher_vector_train_%d.npy' % self.kernel))
                    X_dev = np.load(os.path.join(line, 'fisher_vector_dev_%d.npy' % self.kernel))
                    y_train = np.load(os.path.join(line, 'label_train.npy'))
                    y_dev = np.load(os.path.join(line, 'label_dev.npy'))
                    X_train = np.reshape(X_train, (-1, np.prod(X_train.shape[1:])))
                    X_dev = np.reshape(X_dev, (-1, np.prod(X_dev.shape[1:])))
                    X_train = np.nan_to_num(X_train)
                    X_dev = np.nan_to_num(X_dev)

                    df = pd.DataFrame(np.vstack((X_train, X_dev)))
                    df.columns = ['feature_%d' % i for i in range(len(X_train[0]))]
                    df.insert(0, 'label', np.hstack((y_train, y_dev)).T)
                    print(df.head())

                    feature_list = pymrmr.mRMR(df, 'MIQ', 50)
                    np.save(os.path.join(line, 'feature_list'), feature_list)

                    X_train_df = pd.DataFrame(X_train)
                    X_train_df.columns = ['feature_%d' % i for i in range(len(X_train[0]))]
                    X_train = X_train_df.loc[:, feature_list]

                    X_dev_df = pd.DataFrame(X_dev)
                    X_dev_df.columns = ['feature_%d' % i for i in range(len(X_dev[0]))]
                    X_dev = X_dev_df.loc[:, feature_list]

                    print(X_train.head())
                    print(X_dev.head())

                    np.save(os.path.join(line, 'X_train_mrmr'), X_train)
                    np.save(os.path.join(line, 'X_dev_mrmr'), X_dev)
                    print("\nfeature selection done and data saved.")

    def FV_tree(self):
        print("\nrunning Random Forest algorithm for feature selection")
        ae = AutoEncoder('fv_gmm', 0)

        with smart_open(os.path.join(ae.save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                line = line[:-2]
                print(line_no, '\t', line[19:])

                if os.path.isfile(os.path.join(line, 'X_train_tree_%d.npy' % self.kernel)) and os.path.isfile(os.path.join(line, 'X_dev_tree_%d.npy' % self.kernel)):
                    preprocess_metadata_tensorboard(line, self.kernel)
                    continue

                if os.path.isfile(os.path.join(line, 'fisher_vector_train_%d.npy' % self.kernel)) and os.path.isfile(os.path.join(line, 'fisher_vector_dev_%d.npy' % self.kernel)):
                    X_train = np.load(os.path.join(line, 'fisher_vector_train_%d.npy' % self.kernel))
                    X_dev = np.load(os.path.join(line, 'fisher_vector_dev_%d.npy' % self.kernel))
                    y_train = np.load(os.path.join(line, 'label_train.npy'))
                    y_dev = np.load(os.path.join(line, 'label_dev.npy'))
                    X_train = np.reshape(X_train, (-1, np.prod(X_train.shape[1:])))
                    X_dev = np.reshape(X_dev, (-1, np.prod(X_dev.shape[1:])))
                    X_train = np.nan_to_num(X_train)
                    X_dev = np.nan_to_num(X_dev)

                    from sklearn.ensemble import RandomForestClassifier

                    if not os.path.isfile(os.path.join(line, 'feature_list_%d.npy' % self.kernel)):
                        model = RandomForestClassifier(
                                n_estimators=800,
                                criterion='entropy')

                        df = pd.DataFrame(np.vstack((X_train, X_dev)))
                        feature_names = ['feature_%d' % i for i in range(len(X_train[0]))]
                        df.columns = feature_names
                        y = np.hstack((y_train, y_dev))
                        print(df.head())
                        
                        model.fit(df, y)
                        importances = model.feature_importances_
                        print("\nfeature importance ranking")
                        indices = np.argsort(importances)[::-1]
                        for f in range(100):
                            print("%d. feature %d %s (%f)" % (f+1, indices[f], feature_names[indices[f]], importances[indices[f]]))

                        indices = indices[:100]
                        np.save(os.path.join(line, 'feature_list_%d' % self.kernel), indices)
                    else:
                        indices = np.load(os.path.join(line, 'feature_list_%d.npy' % self.kernel))

                    X_train_df = pd.DataFrame(X_train)
                    X_train_df.columns = ['feature_%d' % i for i in range(len(X_train[0]))]
                    X_train_tree = X_train_df.iloc[:, indices]

                    X_dev_df = pd.DataFrame(X_dev)
                    X_dev_df.columns = ['feature_%d' % i for i in range(len(X_dev[0]))]
                    X_dev_tree = X_dev_df.iloc[:, indices]

                    print(X_train_tree.shape, X_dev_tree.shape)
                    
                    np.save(os.path.join(line, 'X_train_tree_%d' % self.kernel), X_train_tree)
                    np.save(os.path.join(line, 'X_dev_tree_%d' % self.kernel), X_dev_tree)
                    print("\nfeature selection done and data saved.")

    def FV_RF(self):
        print("\nrunning Random Forest on Fisher Vectors")
        ae = AutoEncoder('fv_gmm', 0)

        with smart_open(os.path.join(ae.save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                print(line_no, '\t', line[65:])
                feature_name = line[65:] + '_%d' % self.kernel

                if os.path.isfile(os.path.join(line, 'X_train_tree_%d.npy' % self.kernel)) and os.path.isfile(os.path.join(line, 'X_dev_tree_%d.npy' % self.kernel)):
                    X_train = np.load(os.path.join(line, 'X_train_tree_%d.npy' % self.kernel))
                    X_dev = np.load(os.path.join(line, 'X_dev_tree_%d.npy' % self.kernel))
                    y_train = np.load(os.path.join(line, 'label_train.npy'))
                    y_dev = np.load(os.path.join(line, 'label_dev.npy'))

                    print(X_train.shape, X_dev.shape)

                    random_forest = RandomForest(feature_name, X_train, y_train, X_dev, y_dev, test=False)
                    random_forest.run()
                    y_pred_train, y_pred_dev = random_forest.evaluate()
                    get_UAR(y_pred_train, y_train, np.array([]), 'RF', feature_name, 'single', train_set=True, test=False)
                    get_UAR(y_pred_dev, y_dev, np.array([]), 'RF', feature_name, 'single', test=False)

    def FUSION(self):
        print("\nrunning early fusion strategy on audio-visual-textual modalities")
        model_path_AV = smart_open('./pre-trained/DDAE/model_list.txt', 'rb', encoding='utf-8')
        model_path_T = smart_open('./pre-trained/doc2vec/model_list.txt', 'rb', encoding='utf-8')
        model_list_AV = []
        model_list_T = []

        for _, line_AV in enumerate(model_path_AV):
            line_AV = str(line_AV).replace('\n', '')
            model_list_AV.append(line_AV)
        for _, line_T in enumerate(model_path_T):
            line_T = str(line_T).replace('\n', '')
            model_list_T.append(line_T)

        _, _, y_dev, y_train = load_label()
        y_train = y_train.astype('int')
        y_dev = y_dev.astype('int')
        
        for AV in model_list_AV:
            for T in model_list_T:
                feature_name = AV[19:-2] + T[22:]
                if os.path.isfile(os.path.join('pre-trained', 'fusion', feature_name, 'X_train_tree.npy')) and os.path.isfile(os.path.join('pre-trained', 'fusion', feature_name, 'X_train_tree.npy')):
                    X_train_tree = np.load(os.path.join('pre-trained', 'fusion', feature_name, 'X_train_tree.npy'))
                    X_dev_tree = np.load(os.path.join('pre-trained', 'fusion', feature_name, 'X_train_tree.npy'))
                
                else:
                    X_train_AV = np.load(os.path.join(AV[:-2], 'X_train_tree_%d.npy' % int(AV[-2:])))
                    X_dev_AV = np.load(os.path.join(AV[:-2], 'X_dev_tree_%d.npy' % int(AV[-2:])))
                    X_train_txt = np.load(os.path.join(T, 'vectors_train.npy'))
                    X_dev_txt = np.load(os.path.join(T, 'vectors_dev.npy'))

                    assert X_train_AV.shape[0] == X_train_txt.shape[0] == len(y_train)
                    assert X_dev_AV.shape[0] == X_dev_txt.shape[0] == len(y_dev)

                    X_train = np.hstack((X_train_AV, X_train_txt))
                    X_dev = np.hstack((X_dev_AV, X_dev_txt))

                    os.mkdir(os.path.join('pre-trained', 'fusion', feature_name))
                    np.save(os.path.join('pre-trained', 'fusion', feature_name, 'X_train'), X_train)
                    np.save(os.path.join('pre-trained', 'fusion', feature_name, 'X_dev'), X_dev)

                    from sklearn.ensemble import RandomForestClassifier

                    model = RandomForestClassifier(n_estimators=800, criterion='entropy')

                    df = pd.DataFrame(np.vstack((X_train, X_dev)))
                    feature_names = ['feature_%d' % i for i in range(len(X_train[0]))]
                    df.columns = feature_names
                    y = np.hstack((y_train, y_dev))
                    
                    model.fit(df, y)
                    importances = model.feature_importances_
                    print("\nfeature importance ranking")
                    indices = np.argsort(importances)[::-1]
                    for f in range(100):
                        print("%d. feature %d %s (%f)" % (f+1, indices[f], feature_names[indices[f]], importances[indices[f]]))
                    
                    indices = indices[:100]
                    np.save(os.path.join('pre-trained', 'fusion', feature_name, 'feature_list'), indices)

                    X_train_df = pd.DataFrame(X_train)
                    X_train_df.columns = ['feature_%d' % i for i in range(len(X_train[0]))]
                    X_train_tree = X_train_df.iloc[:, indices]

                    X_dev_df = pd.DataFrame(X_dev)
                    X_dev_df.columns = ['feature_%d' % i for i in range(len(X_dev[0]))]
                    X_dev_tree = X_dev_df.iloc[:, indices]

                    np.save(os.path.join('pre-trained', 'fusion', feature_name, 'X_train_tree'), X_train_tree)
                    np.save(os.path.join('pre-trained', 'fusion', feature_name, 'X_dev_tree'), X_dev_tree)

    def DNN(self):
        print("\nrunning Multi-Task DNN on features selected with RF with doc2vec embeddings")
        
        feature_path = smart_open('./pre-trained/fusion/feature_list.txt', 'rb', encoding='utf-8')
        feature_list = []
        for _, line in enumerate(feature_path):
            feature_list.append(str(line).replace('\n', ''))
        
        feature = feature_list[0]

        X_train = np.load(os.path.join('pre-trained', 'fusion', feature, 'X_train_tree.npy'))
        X_dev = np.load(os.path.join('pre-trained', 'fusion', feature, 'X_dev_tree.npy'))

        y_dev_r, y_train_r, y_dev, y_train = load_label()
        y_train = y_train.astype('int')
        y_dev = y_dev.astype('int')
        num_classes = 3
        
        if False:
            multi_dnn = MultiTaskDNN(feature, X_train.shape[1], num_classes)
            multi_dnn.build_model()
            multi_dnn.train_model(X_train, y_train, y_train_r, X_dev, y_dev, y_dev_r)
            multi_dnn.evaluate_model(X_train, y_train, y_train_r, X_dev, y_dev, y_dev_r)
        else:
            single_dnn = SingleTaskDNN(feature, X_train.shape[1], num_classes)
            single_dnn.build_model()
            single_dnn.train_model(X_train, y_train, X_dev, y_dev)
            single_dnn.evaluate_model(X_dev, y_dev)
    
    def RF(self):
        print("\nrunning RF on features selected with RF with doc2vec embeddings")
        
        feature_path = smart_open('./pre-trained/fusion/feature_list.txt', 'rb', encoding='utf-8')
        feature_list = []
        for _, line in enumerate(feature_path):
            feature_list.append(str(line).replace('\n', ''))
        
        for _ in range(3):
            for feature in feature_list:
                _, _, y_dev, y_train = load_label()
                y_train = y_train.astype('int')
                y_dev = y_dev.astype('int')

                X_train = np.load(os.path.join('pre-trained', 'fusion', feature, 'X_train.npy'))
                X_dev = np.load(os.path.join('pre-trained', 'fusion', feature, 'X_dev.npy'))

                random_forest = RandomForest(feature, X_train, y_train, X_dev, y_dev, test=False)
                random_forest.run()
                y_pred_train, y_pred_dev = random_forest.evaluate()
                get_UAR(y_pred_train, y_train, np.array([]), 'RF', feature, 'multiple', train_set=True, test=False)
                get_UAR(y_pred_dev, y_dev, np.array([]), 'RF', feature, 'multiple', test=False)
        
    def RF_CV(self):
        print("\nrunning RF on features selected with RF with doc2vec embeddings")
        
        feature_path = smart_open('./pre-trained/fusion/feature_list.txt', 'rb', encoding='utf-8')
        feature_list = []
        for _, line in enumerate(feature_path):
            feature_list.append(str(line).replace('\n', ''))
        
        from sklearn.metrics import precision_recall_fscore_support

        cv_results_UAR = dict()
        cv_results_UAP = dict()
        
        for feature in feature_list:
            cv_results_UAR[feature] = []
            cv_results_UAP[feature] = []

            _, _, y_dev, y_train = load_label()
            y_train = y_train.astype('int')
            y_dev = y_dev.astype('int')

            X_train = np.load(os.path.join('pre-trained', 'fusion', feature, 'X_train.npy'))
            X_dev = np.load(os.path.join('pre-trained', 'fusion', feature, 'X_dev.npy'))

            X = np.vstack((X_train, X_dev))
            y = np.hstack((y_train, y_dev))

            cv_ids = k_fold_cv(len(X))

            for cv_id in cv_ids:
                X_train = X[cv_id[0]]
                y_train = y[cv_id[0]]
                X_dev = X[cv_id[1]]
                y_dev = y[cv_id[1]]

                print('train on %d test on %d' % (len(y_train), len(y_dev)))

                random_forest = RandomForest(feature, X_train, y_train, X_dev, y_dev, test=False)
                random_forest.run()
                _, y_pred = random_forest.evaluate()
                precision, recall, _, _ = precision_recall_fscore_support(y_dev, y_pred, average='macro')
                cv_results_UAR[feature].append(recall)
                cv_results_UAP[feature].append(precision)
            
            assert len(cv_results_UAR[feature]) == len(cv_results_UAP[feature]) == 10

        with open(os.path.join('results', 'cross-validation.json'), 'a+', encoding='utf-8') as outfile:
            json.dump(cv_results_UAR, outfile)
            json.dump(cv_results_UAP, outfile)

    def TEXT(self):
        print("\nrunning doc2vec embeddings on text modality")
        text2vec = Text2Vec(build_on_corpus=True)
        text2vec.build_model()
        text2vec.train_model()
        text2vec.infer_embedding('train')
        text2vec.infer_embedding('dev')

    def TEXT_RF(self):
        print("\nrunning Random Forest on document embeddings")

        text2vec = Text2Vec()

        with smart_open(os.path.join(text2vec.model_config['doc2vec']['save_dir'], 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                print(line_no, '\t', line[68:])
                X_train = np.load(os.path.join(line, 'vectors_train.npy'))
                X_dev = np.load(os.path.join(line, 'vectors_dev.npy'))
                y_train = np.load(os.path.join(line, 'labels_train.npy'))
                y_dev = np.load(os.path.join(line, 'labels_dev.npy'))
                y_train = np.ravel(y_train)
                y_dev = np.ravel(y_dev)
                random_forest = RandomForest(line[68:], X_train, y_train, X_dev,y_dev, baseline=False)
                random_forest.run()
                y_pred_train, y_pred_dev = random_forest.evaluate()
                get_UAR(y_pred_train, y_train, np.array([]), 'RF', line[68:], 'single', baseline=False, train_set=True)
                get_UAR(y_pred_dev, y_dev, np.array([]), 'RF', line[68:], 'single', baseline=False)

    def PERMUTATION(self):
        test = PermutationTest(6,4, R=5000, baseline=False)
        test.run()