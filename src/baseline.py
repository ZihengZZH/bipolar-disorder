import json
import math
import numpy as np
import pandas as pd
import scipy
from src.utility import load_MATLAB_baseline_feature, load_label
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import multiprocessing


'''
BASELINE CLASSIFICATION (py) PROVIDED BY AVEC2018
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))
model_config = json.load(open('./config/model.json', 'r'))
N_JOBS = multiprocessing.cpu_count()


class BaseLine():
    def __init__(self, name='ALL'):
        self.name = name
        self.train_len = None
        self.dev_len = None
        # self.test_len = None
        self._load_basics()
        # self.run()

    def _load_basics(self):
        self.train_len = int(data_config['train_len'])
        self.dev_len = int(data_config['dev_len'])
        # self.test_len = data_config['test_len']

    def run(self):
        if self.name == 'ALL':
            self.run_BoAW()
            self.run_MFCC()
            self.run_eGeMAPS()
            self.run_DeepSpectrum()
            self.run_BoVW()
            self.run_AU()
            self.fusion()
        elif self.name == 'BoAW':
            self.run_BoAW()
        elif self.name == 'MFCC':
            self.run_MFCC()
        elif self.name == 'eGeMAPS':
            self.run_eGeMAPS()
        elif self.name == 'Deep':
            self.run_DeepSpectrum()
        elif self.name == 'BoVW':
            self.run_BoVW()
        elif self.name == 'AU':
            self.run_AU()
    
    def run_BoAW(self):
        return 0

    def run_MFCC(self):
        X_train, y_train, train_inst, X_test, y_test, test_inst = load_MATLAB_baseline_feature('MFCC')
        
        y_pred = self.run_Random_Forest(X_train, y_train, X_test, y_test)
        self.get_UAR(y_pred, y_test, test_inst)
        # self.run_Random_Forest(feature_all.values, label_all.T.values)
        return 0

    def run_eGeMAPS(self):
        return 0

    def run_DeepSpectrum(self):
        X_train, y_train, train_inst, X_test, y_test, test_inst = load_MATLAB_baseline_feature('Deep')

        y_pred = self.run_Random_Forest(X_train, y_train, X_test, y_test)
        self.get_UAR(y_pred, y_test, test_inst)
        # self.tune_parameters_Random_Forest(feature_all.values, label_all.T.values)

    def run_BoVW(self):
        return 0

    def run_AU(self):
        return 0

    def run_Random_Forest(self, X_train, y_train, X_test, y_test):
        # X_train, y_train, train_inst, X_test, y_test, test_inst = load_MATLAB_baseline_feature('Deep')
        # X_train, X_test, y_train, y_test = train_test_split(features, labels)
        y_train, y_test = y_train.T.values, y_test.T.values

        print("\ntraining a Random Forest Classifier ...")
        forest = RandomForestClassifier(n_estimators=200, max_features=0.5, criterion='entropy', verbose=1, n_jobs=N_JOBS)
        forest.fit(X_train, y_train)

        print("\ntesting the Random Forest Classifier ...")
        print("\naccuracy on training set: %.3f" % forest.score(X_train, y_train))
        print("\naccuracy on test set: %.3f" % forest.score(X_test, y_test))

        y_pred = forest.predict(X_test)
        return y_pred

    def tune_parameters_Random_Forest(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels)
        parameters = {
            "n_estimators": model_config['random_forest']['n_estimators'],
            "max_features": model_config['random_forest']['max_features'],
            "criterion": model_config['random_forest']['criterion']
        }

        print("\nrunning the Grid Search for Random Forest classifier ...")
        clf = GridSearchCV(RandomForestClassifier(), parameters, cv=3, n_jobs=N_JOBS, verbose=10)

        clf.fit(X_train, y_train)
        print(clf.score(X_train, y_train))
        print(clf.best_params_)
        print(clf.cv_results_['mean_test_score'])
        print(clf.cv_results_['std_test_score'])

    def fusion(self):
        return 0

    # get UAR metric for both frame-level and session-level
    def get_UAR(self, y_pred, y_test, inst, frame=True, session=True):
        # para y_pred: predicted mania level for each frame
        # para y_test: actual mania level for each frame
        # para inst: session mappings of frames
        if frame:
            # get recalls for three classes
            recall = [0] * 3
            for i in range(3):
                index, = np.where(y_test == (i+1))
                index_pred, = np.where(y_pred[index] == (i+1))
                recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
            print("\nUAR (mean of recalls) based on frame-level is ", np.mean(recall))
        
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
            print("\nUAR (mean of recalls) based on session-level is ", np.mean(recall))
