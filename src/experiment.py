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
                    'DDAE on unimodality', 'BiDDAE on aligned A/V',
                    'MultiDDAE on aligned A/V',
                    'Dynamics on latent repres',
                    'FV using GMM on latent repres', 
                    'RF on FV',
                    'DNN as classifier', 'doc2vec on text',
                    'RF on doc2vec']
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
            self.DYNAMICS()
        elif choice == 5:
            self.FV_GMM()
        elif choice == 6:
            self.FV_RF()
        elif choice == 7:
            self.DNN()
        elif choice == 8:
            self.TEXT()
        elif choice == 9:
            self.TEXT_RF()
    
    def main_system(self):
        print("\nrunning the proposed architecture (BiDDAE + FV + DNN)")

    def BAE_BOXW(self):
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

    def AE_single(self):
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
    
    def BAE_bimodal(self):
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

    def BAEV_multimodal(self):
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
        print("\nrunning computation of dynamics of latent reprsentation learned by DDAEs")
        import os
        from smart_open import smart_open

        with smart_open('/media/zzh/Ziheng-700G/Dataset/bipolar-disorder/pre-trained/DDAE/model_list.txt', 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                print(line_no, '\t', line)
                
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

    def FV_GMM(self):
        print("\nrunning Fisher Encoder using GMM on learnt representations with dynamics")
        import os
        from smart_open import smart_open

        y_train_frame, inst_train, y_dev_frame, inst_dev = load_aligned_features(no_data=True, verbose=True)
        y_train_frame, y_dev_frame = y_train_frame[2:,:], y_dev_frame[2:,:]
        inst_train, inst_dev = inst_train[2:,:], inst_dev[2:,:]

        ae = AutoEncoder('fv_gmm', 0)

        with smart_open(os.path.join(ae.save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                print(line_no, '\t', line[65:])

                if os.path.isfile(os.path.join(line, 'fisher_vector_train_64_norm.npy')) and os.path.isfile(os.path.join(line, 'fisher_vector_dev_64_norm.npy')):
                    continue
                
                X_train_frame = np.load(os.path.join(line, 'encoded_train_dynamics.npy'))
                X_dev_frame = np.load(os.path.join(line, 'encoded_dev_dynamics.npy'))

                fv_gmm = FisherVectorGMM(n_kernels=64)
                fv_gmm.fit(np.vstack((np.vstack(X_train_frame), np.vstack(X_dev_frame))))

                X_train_session, y_train_session = frame2session(X_train_frame, y_train_frame, inst_train, verbose=True)
                X_dev_session, y_dev_session = frame2session(X_dev_frame, y_dev_frame, inst_dev, verbose=True)

                fv_train = np.array([fv_gmm.predict(train) for train in X_train_session])
                fv_dev = np.array([fv_gmm.predict(dev) for dev in X_dev_session])

                fv_gmm.data_dir = line

                fv_gmm.save_vector(fv_train, 'train')
                fv_gmm.save_vector(fv_dev, 'dev')
                fv_gmm.save_vector(y_train_session, 'train', label=True)
                fv_gmm.save_vector(y_dev_session, 'dev', label=True)
                print("\nFV encoding for %s, done" % line[65:])
    
    def FV_RF(self):
        print("\nrunning Random Forest on Fisher Vectors")
        import os
        from smart_open import smart_open

        ae = AutoEncoder('fv_gmm', 0)

        with smart_open(os.path.join(ae.save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
            for line_no, line in enumerate(model_path):
                line = str(line).replace('\n', '')
                print(line_no, '\t', line[65:])
                feature_name = line[65:] + '_64_norm'

                if os.path.isfile(os.path.join(line, 'fisher_vector_train_64_norm.npy')) and os.path.isfile(os.path.join(line, 'fisher_vector_dev_64_norm.npy')):
                    X_train = np.load(os.path.join(line, 'fisher_vector_train_64_norm.npy'))
                    X_dev = np.load(os.path.join(line, 'fisher_vector_dev_64_norm.npy'))
                    y_train = np.load(os.path.join(line, 'label_train.npy'))
                    y_dev = np.load(os.path.join(line, 'label_dev.npy'))
                    X_train = np.reshape(X_train, (-1, np.prod(X_train.shape[1:])))
                    X_dev = np.reshape(X_dev, (-1, np.prod(X_dev.shape[1:])))
                    X_train = np.nan_to_num(X_train)
                    X_dev = np.nan_to_num(X_dev)

                    X_train, y_train, _ = upsample(X_train, y_train, np.array([]))
                    
                    random_forest = RandomForest(feature_name, X_train, y_train, X_dev, y_dev, baseline=False)
                    random_forest.run()
                    y_pred_train, y_pred_dev = random_forest.evaluate()
                    get_UAR(y_pred_train, y_train, np.array([]), 'RF', feature_name, 'single', baseline=False, train_set=True)
                    get_UAR(y_pred_dev, y_dev, np.array([]), 'RF', feature_name, 'single', baseline=False)

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

    def TEXT_RF(self):
        print("\nrunning Random Forest on document embeddings")
        import os
        from smart_open import smart_open

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