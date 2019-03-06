import json
import math
import numpy
import pandas
import scipy
from src.utility import load_MATLAB_label, load_MATLAB_baseline_feature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import multiprocessing


'''
BASELINE CLASSIFICATION (py) PROVIDED BY AVEC2018
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))
N_JOBS = multiprocessing.cpu_count() * 3


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
        feature_all, label_all = load_MATLAB_baseline_feature('MFCC'), load_MATLAB_label('MFCC')
        
        self.run_Random_Forest(feature_all.values, label_all.T.values)

    def run_eGeMAPS(self):
        return 0

    def run_DeepSpectrum(self):
        feature_all, label_all = load_MATLAB_baseline_feature('Deep'), load_MATLAB_label('Deep')

        self.run_Random_Forest(feature_all.values, label_all.T.values)

    def run_BoVW(self):
        return 0

    def run_AU(self):
        return 0

    def run_Random_Forest(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels)

        print("\ntraining a Random Forest Classifier ...")
        forest = RandomForestClassifier(n_estimators=500, criterion='entropy', verbose=1, n_jobs=N_JOBS)
        forest.fit(X_train, y_train)

        print("\ntesting the Random Forest Classifier ...")
        print("\naccuracy on training set: %.3f" % forest.score(X_train, y_train))
        print("\naccuracy on test set: %.3f" % forest.score(X_test, y_test))


    def fusion(self):
        return 0