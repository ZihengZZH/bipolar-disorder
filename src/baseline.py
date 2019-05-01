import json
import math
import numpy as np
import pandas as pd
import scipy
from src.utility import load_proc_baseline_feature, load_label, save_results
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
        print("\nbaseline system initialized")

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
            self.fusion()
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
        X_train, y_train, train_inst, X_test, y_test, dev_inst = load_proc_baseline_feature('MFCC', verbose=True)
        
        y_pred = self.run_Random_Forest(X_train, y_train, X_test, y_test)
        self.get_UAR(y_pred, np.ravel(y_test), np.ravel(dev_inst), 'MFCC')

    def run_eGeMAPS(self):
        print("\nbuilding a classifier on eGeMAPS features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_test, y_test, dev_inst = load_proc_baseline_feature('eGeMAPS', verbose=True)
        
        y_pred = self.run_Random_Forest(X_train, y_train, X_test, y_test)
        self.get_UAR(y_pred, np.ravel(y_test), np.ravel(dev_inst), 'eGeMAPS')

    def run_DeepSpectrum(self):
        print("\nbuilding a classifier on Deep features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_test, y_test, dev_inst = load_proc_baseline_feature('Deep', verbose=True)

        y_pred = self.run_Random_Forest(X_train, y_train, X_test, y_test)
        self.get_UAR(y_pred, np.ravel(y_test), np.ravel(dev_inst), 'Deep')

    def run_BoAW(self):
        print("\nbuilding a classifier on BoAW features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_test, y_test, dev_inst = load_proc_baseline_feature('BoAW', verbose=True)

        y_pred = self.run_Random_Forest(X_train, y_train, X_test, y_test)
        self.get_UAR(y_pred, np.ravel(y_test), np.ravel(dev_inst), 'BoAW')

    def run_AU(self):
        print("\nbuilding a classifier on AU features (both frame-level and session-level)")
        X_train, y_train, _, X_test, y_test, _ = load_proc_baseline_feature('AU', verbose=True)

        y_pred = self.run_Random_Forest(X_train, y_train, X_test, y_test)
        self.get_UAR(y_pred, np.ravel(y_test), np.array([]), 'AU')

    def run_BoVW(self):
        print("\nbuilding a classifier on BoVW features (already session-level)")
        X_train, y_train, train_inst, X_test, y_test, dev_inst = load_proc_baseline_feature('BoVW', verbose=True)

        y_pred = self.run_Random_Forest(X_train, y_train, X_test, y_test)
        self.get_UAR(y_pred, np.ravel(y_test), np.ravel(dev_inst), 'BoVW')

    def run_linear_SVM(self, X_train, y_train, X_test, y_test):
        pass

    def run_Random_Forest(self, X_train, y_train, X_test, y_test):
        y_train, y_test = y_train.T.values, y_test.T.values

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
        print("\naccuracy on development set: %.3f" % forest.score(X_test, np.ravel(y_test)))

        y_pred = forest.predict(X_test)
        return y_pred

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

    def fusion(self):
        return 0

    # get UAR metric for both frame-level and session-level
    def get_UAR(self, y_pred, y_test, inst, frame=True, session=True):
        # para y_pred: predicted mania level for each frame
        # para y_test: actual mania level for each frame
        # para inst: session mappings of frames
        frame_res, session_res = 0.0, 0.0

        # UAR for session-level only (AU features)
        if not inst.any():
            # get recalls for three classes
            recall = [0] * 3
            for i in range(3):
                index, = np.where(y_test == (i+1))
                index_pred, = np.where(y_pred[index] == (i+1))
                recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
            session_res = np.mean(recall)
            print("\nUAR (mean of recalls) using %s feature based on session-level is %.2f" % (self.name, session_res))
            save_results(frame_res, session_res, self.name, 'single')
        
        else:
            # UAR for frame-level
            if frame:
                # get recalls for three classes
                recall = [0] * 3
                for i in range(3):
                    index, = np.where(y_test == (i+1))
                    index_pred, = np.where(y_pred[index] == (i+1))
                    recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
                frame_res = np.mean(recall)
                print("\nUAR (mean of recalls) using %s feature based on frame-level is %.2f" % (self.name, frame_res))
            
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
                print("\nUAR (mean of recalls) using %s feature based on session-level is %.2f" % (self.name, session_res))

            save_results(frame_res, session_res, self.name, 'single')
