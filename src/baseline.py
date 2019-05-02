import json
import math
import numpy as np
import pandas as pd
import scipy
from src.utility import load_proc_baseline_feature, load_label, save_results
from src.utility import save_post_probability, load_post_probability
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import multiprocessing


'''
BASELINE CLASSIFICATION (py) PROVIDED BY AVEC2018

features        | computed level 
--------        | --------------
MFCCs           | frame level
eGeMAPS         | turn level
BoAW            | window size (2s)
Deep Spectrum   | activations in ALEXNET
FAUs            | session level
BoVW            | window size (11s)

'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))
model_config = json.load(open('./config/model.json', 'r'))
N_JOBS = multiprocessing.cpu_count()


class BaseLine():
    def __init__(self, model, name, test=False):
        # para model: determine the model in baseline system
        # para name: determine the feature in baseline system
        if model == 'SVM' or model == 'RF':
            self.model = model
        self.name = name
        self.test = test
        self.length_train = None
        self.length_dev = None
        self.parameters_RF = dict()
        self.parameters_SVM = dict()
        self._load_basics()
        self.session_prob = None # for AU feature only
        print("\nbaseline system initialized, feature %s" % name)

    def _load_basics(self):
        self.length_train = int(data_config['length_train'])
        self.length_dev = int(data_config['length_dev'])
        if self.test:
            self.parameters_RF['n_estimators'] = 100
            self.parameters_RF['max_features'] = 0.1
            self.parameters_RF['max_depth'] = 4
            self.parameters_RF['criterion'] = 'entropy'
            self.parameters_SVM['C'] = 10
        else:
            self.parameters_RF['n_estimators'] = None
            self.parameters_RF['max_features'] = None
            self.parameters_RF['max_depth'] = None
            self.parameters_RF['criterion'] = None
            self.parameters_SVM['C'] = None

    def run(self):
        if self.name == 'ALL':
            self.run_MFCC()
            self.run_eGeMAPS()
            self.run_DeepSpectrum()
            self.run_BoAW()
            self.run_AU()
            self.run_BoVW()
        elif self.name == 'MFCC':
            self.run_MFCC()
        elif self.name == 'eGeMAPS':
            self.run_eGeMAPS()
        elif self.name == 'Deep':
            self.run_DeepSpectrum()
        elif self.name == 'BoAW':
            self.run_BoAW()
        elif self.name == 'AU':
            self.run_AU()
        elif self.name == 'BoVW':
            self.run_BoVW()

    def run_MFCC(self):
        print("\nbuilding a classifier on MFCC features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('MFCC', verbose=True)
        
        y_pred_train, y_pred_dev = self.run_Random_Forest(X_train, y_train, X_dev, y_dev)
        self.get_UAR(y_pred_train, np.ravel(y_train), np.ravel(train_inst), train_set=True)
        self.get_UAR(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))
        self.get_post_probability(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))

    def run_eGeMAPS(self):
        print("\nbuilding a classifier on eGeMAPS features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('eGeMAPS', verbose=True)
        
        y_pred_train, y_pred_dev = self.run_Random_Forest(X_train, y_train, X_dev, y_dev)
        self.get_UAR(y_pred_train, np.ravel(y_train), np.ravel(train_inst), train_set=True)
        self.get_UAR(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))
        self.get_post_probability(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))

    def run_DeepSpectrum(self):
        print("\nbuilding a classifier on Deep features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('Deep', verbose=True)

        y_pred_train, y_pred_dev = self.run_Random_Forest(X_train, y_train, X_dev, y_dev)
        self.get_UAR(y_pred_train, np.ravel(y_train), np.ravel(train_inst), train_set=True)
        self.get_UAR(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))
        self.get_post_probability(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))

    def run_BoAW(self):
        print("\nbuilding a classifier on BoAW features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('BoAW', verbose=True)

        y_pred_train, y_pred_dev = self.run_Random_Forest(X_train, y_train, X_dev, y_dev)
        self.get_UAR(y_pred_train, np.ravel(y_train), np.ravel(train_inst), train_set=True)
        self.get_UAR(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))
        self.get_post_probability(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))

    def run_AU(self):
        print("\nbuilding a classifier on AU features (already session-level)")
        X_train, y_train, _, X_dev, y_dev, _ = load_proc_baseline_feature('AU', verbose=True)

        y_pred_train, y_pred_dev = self.run_Random_Forest(X_train, y_train, X_dev, y_dev)
        self.get_UAR(y_pred_train, np.ravel(y_train), np.ravel([]), train_set=True)
        self.get_UAR(y_pred_dev, np.ravel(y_dev), np.ravel([]))
        self.get_post_probability(y_pred_dev, np.ravel(y_dev), np.ravel([]))

    def run_BoVW(self):
        print("\nbuilding a classifier on BoVW features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('BoVW', verbose=True)

        y_pred_train, y_pred_dev = self.run_Random_Forest(X_train, y_train, X_dev, y_dev)
        self.get_UAR(y_pred_train, np.ravel(y_train), np.ravel(train_inst), train_set=True)
        self.get_UAR(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))
        self.get_post_probability(y_pred_dev, np.ravel(y_dev), np.ravel(dev_inst))

    def run_linear_SVM(self, X_train, y_train, X_dev, y_dev):
        pass

    def run_Random_Forest(self, X_train, y_train, X_dev, y_dev):
        y_train, y_dev = y_train.T.values, y_dev.T.values

        if not self.parameters_RF['n_estimators'] or not self.parameters_RF['max_features'] or not self.parameters_RF['max_depth'] or not self.parameters_RF['criterion']:
            print("\nhyperparameters are not tuned yet")
            self.tune_parameters_Random_Forest(X_train, np.ravel(y_train))
        else:
            print("\nno fine-tunning this time")

        print("\ntraining a Random Forest Classifier ...")
        forest = RandomForestClassifier(n_estimators=self.parameters_RF['n_estimators'], max_features=self.parameters_RF['max_features'], max_depth=self.parameters_RF['max_depth'] , criterion=self.parameters_RF['criterion'], verbose=1, n_jobs=N_JOBS)
        forest.fit(X_train, np.ravel(y_train))

        print("\ntesting the Random Forest Classifier ...")
        print("\naccuracy on training set: %.3f" % forest.score(X_train, np.ravel(y_train)))
        print("\naccuracy on development set: %.3f" % forest.score(X_dev, np.ravel(y_dev)))
        
        y_pred_train = forest.predict(X_train)
        y_pred_dev = forest.predict(X_dev)

        if self.name == 'AU':
            self.session_prob = forest.predict_proba(X_dev)

        return y_pred_train, y_pred_dev

    def tune_parameters_Random_Forest(self, data, labels):
        # para data: training data to tune the classifier
        # para labels: training labels to tune the classifier
        parameters = {
            "n_estimators": model_config['baseline']['random_forest']['n_estimators'],
            "max_features": model_config['baseline']['random_forest']['max_features'],
            "max_depth": model_config['baseline']['random_forest']['max_depth'],
            "criterion": model_config['baseline']['random_forest']['criterion']
        }

        print("\nrunning the Grid Search for Random Forest classifier ...")
        clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, n_jobs=N_JOBS, verbose=1)

        clf.fit(data, labels)
        print(clf.score(data, labels))
        print(clf.best_params_)
        print(clf.cv_results_['mean_test_score'])
        print(clf.cv_results_['std_test_score'])

        self.parameters_RF['n_estimators'] = clf.best_params_['n_estimators']
        self.parameters_RF['max_features'] = clf.best_params_['max_features']
        self.parameters_RF['max_depth'] = clf.best_params_['max_depth']
        self.parameters_RF['criterion'] = clf.best_params_['criterion']

        # write to model json file
        with open('./config/model.json', 'a+') as output:
            json.dump(clf.best_params_, output)
            output.write("\n")
        output.close()

    # apply late fusion strategy on posterior probabilities of two modalities
    def fusion(self, feature_1, feature_2):
        # para feature_1: 1st of fused representations
        # para feature_2: 2nd of fused representations
        prob_dev_1 = load_post_probability(feature_1)
        prob_dev_2 = load_post_probability(feature_2)
        
        assert prob_dev_1.shape == prob_dev_2.shape
        """
        PROB_DEV_1 = (3, 60)
        PROB_DEV_2 = (3, 60)
        """

        _, _, level_dev, level_train = load_label()
        y_dev = level_dev.values[:,1]
        # get the shape
        (_, num_inst) = prob_dev_1.shape
        y_pred = np.array([0] * num_inst)

        for i in range(num_inst):
            prob = prob_dev_1[:,i] + prob_dev_2[:,i]
            # fusion based on majority voting and averaging two modalities
            y_pred[i] = np.argmax(prob) + 1

        self.get_UAR(y_pred, y_dev, np.array([]), fusion=True)

    # get UAR metric for both frame-level and session-level
    def get_UAR(self, y_pred, y_dev, inst, frame=True, session=True, train_set=False, fusion=False):
        # para y_pred: predicted mania level for each frame
        # para y_dev: actual mania level for each frame
        # para inst: session mappings of frames
        # para frame:
        # para session:
        # para train_set:
        # para fusion:
        frame_res, session_res = 0.0, 0.0

        # UAR for session-level only (AU features)
        if not inst.any():
            # get recalls for three classes
            recall = [0] * 3
            for i in range(3):
                index, = np.where(y_dev == (i+1))
                index_pred, = np.where(y_pred[index] == (i+1))
                recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
            session_res = np.mean(recall)
            if not fusion:
                if train_set:
                    print("\nUAR (mean of recalls) using %s feature based on session-level (training set) is %.2f" % (self.name, session_res))
                else:
                    print("\nUAR (mean of recalls) using %s feature based on session-level (development set) is %.2f" % (self.name, session_res))
                    save_results(frame_res, session_res, self.name, 'single')
            else:
                print("\nUAR (mean of recalls) using fusion based on session-level is %.2f" % session_res)
                save_results(frame_res, session_res, 'fusion', 'multiple')

        else:
            # UAR for frame-level
            if frame:
                # get recalls for three classes
                recall = [0] * 3
                for i in range(3):
                    index, = np.where(y_dev == (i+1))
                    index_pred, = np.where(y_pred[index] == (i+1))
                    recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
                frame_res = np.mean(recall)
                if train_set:
                    print("\nUAR (mean of recalls) using %s feature based on frame-level (training set) is %.2f" % (self.name, frame_res))
                else:
                    print("\nUAR (mean of recalls) using %s feature based on frame-level (development set) is %.2f" % (self.name, frame_res))
            
            # UAR for session-level
            if session:
                # get majority-voting for each session
                decision = np.array(([0] * inst.max()))
                for j in range(len(decision)):
                    index, = np.where(inst == (j+1))
                    count = [0] * 3
                    for k in range(3):
                        index_pred, = np.where(y_pred[index] == (k+1))
                        count[k] = len(index_pred)
                    decision[j] = np.argmax(count) + 1

                # get recalls for three classes
                recall = [0] * 3
                _, _, level_dev, level_train = load_label()
                labels = level_dev.iloc[:, 1].tolist()
                labels = np.array(labels, dtype=np.int8)
                for i in range(3):
                    index, = np.where(labels == (i+1))
                    index_pred, = np.where(decision[index] == (i+1))
                    recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
                session_res = np.mean(recall)
                if train_set:
                    print("\nUAR (mean of recalls) using %s feature based on session-level (training set) is %.2f" % (self.name, session_res))
                else:
                    print("\nUAR (mean of recalls) using %s feature based on session-level (development set) is %.2f" % (self.name, session_res))
            
            if not train_set:
                save_results(frame_res, session_res, self.name, 'single')

    # get posteriors probabilities for features
    def get_post_probability(self, y_pred, y_test, inst):
        # para y_pred: predicted mania level for each frame
        # para y_dev: actual mania level for each frame
        # para inst: session mappings of frames
        if self.name != 'AU':
            len_inst = inst.max()
            prob_dev = np.zeros((3, len_inst))
            # assign values
            for l in range(len_inst):
                index, = np.where(inst == (l+1))
                len_index = len(index)
                for n in range(3):
                    index_pred, = np.where(y_pred[index] == (n+1))
                    prob_dev[n][l] = len(index_pred) / len_index
        else:
            if self.session_prob.any():
                prob_dev = self.session_prob.T

        save_post_probability(prob_dev, self.name)