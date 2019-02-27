import numpy
import pandas
import scipy
import src.utility as ut


'''
BASELINE CLASSIFICATION (py) PROVIDED BY AVEC2018
'''


class BaseLine():
    def __init__(self, name):
        self.name = name
        self.run()

    def run(self):
        if self.name == 'MFCC':
            self.run_MFCC()
        elif self.name == 'eGeMAPS':
            self.run_eGeMAPS()
    
    def run_MFCC(self):
        return 0

    def run_eGeMAPS(self):
        feature_all = []
        label_all = []

        label_train, label_dev = ut.load_label()

        for i in range(1, 105):
            feature = ut.load_baseline_feature('eGeMAPS', 'train', i)
            feature_flat = feature.flatten()

            feature_all.append(feature_flat)
            label_all.append(label_train.iloc[i, 1])
