import os
import json
import numpy as np

from src.model.linear_svm import LinearSVM
from src.model.random_forest import RandomForest
from src.metric.uar import get_UAR, get_post_probability, get_late_fusion_UAR
from src.utility.io import load_proc_baseline_feature, load_label, save_UAR_results
from src.utility.io import save_post_probability, load_post_probability


'''
BASELINE CLASSIFICATION (py) PROVIDED BY AVEC2018

features        | computed level 
--------        | --------------
MFCCs           | frame level
eGeMAPS         | turn level
Deep Spectrum   | activations in ALEXNET
BoAW            | window size (2s)
FAUs            | session level
BoVW            | window size (11s)
'''


class BaseLine():
    """
    Baseline system in BD classification, based on SVM/RF using LLDs and fusion
    ---
    Attributes
    -----------
    model_name: str
        model for BaseLine() instance, SVM or RF
    feature_name: str
        feature for BaseLine() instance, MFCC/eGeMAPS/Deep/BoAW/FAU/BoVW
    test: bool
        whether to test BaseLine() or not
    ----------------------------------------------------------------------
    Functions
    -----------
    run(): public
        main function
    run_[MFCC,eGeMAPS,DeepSpectrum,BoAW,AU,BoVW](): public
        run classifier on specified feature (single modality)
    run_fusion(): public
        run late fusion on a pair of specified features
    """
    def __init__(self, model_name, feature_name, test=False):
        # para model: determine the model in baseline system
        # para name: determine the feature in baseline system
        self.model_name = model_name
        self.feature_name = feature_name
        self.test = test
        print("\nbaseline system initialized, model %s feature %s" % (self.model_name, self.feature_name))

    def run(self):
        """main function of BaseLine() instance
        """
        if self.feature_name == 'FUSE':
            feature_name_1 = ''
            feature_name_2 = ''
            self.run_fusion(feature_name_1, feature_name_2)
        elif self.feature_name == 'MFCC':
            self.run_MFCC()
        elif self.feature_name == 'eGeMAPS':
            self.run_eGeMAPS()
        elif self.feature_name == 'Deep':
            self.run_DeepSpectrum()
        elif self.feature_name == 'BoAW':
            self.run_BoAW()
        elif self.feature_name == 'AU':
            self.run_AU()
        elif self.feature_name == 'BoVW':
            self.run_BoVW()

    def run_MFCC(self):
        """run classifier on MFCC feature (single modality)
        """
        print("\nbuilding a classifier on MFCC features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('MFCC', verbose=True)

        if self.model_name == 'SVM':
            SVM_MFCC = LinearSVM(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            SVM_MFCC.run()
            y_pred_train, y_pred_dev = SVM_MFCC.evaluate()
        elif self.model_name == 'RF':
            RF_MFCC = RandomForest(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            RF_MFCC.run()
            y_pred_train, y_pred_dev = RF_MFCC.evaluate()
        
        get_UAR(y_pred_train, y_train, train_inst, self.model_name, self.feature_name, 'baseline', baseline=True, train_set=True, test=self.test)
        get_UAR(y_pred_dev, y_dev, dev_inst, self.model_name, self.feature_name, 'baseline', baseline=True, test=self.test)
        if not self.test:
            get_post_probability(y_pred_dev, y_dev, dev_inst, np.array([]), self.model_name, self.feature_name)

    def run_eGeMAPS(self):
        """run classifier on eGeMAPS feature (single modality)
        """
        print("\nbuilding a classifier on eGeMAPS features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('eGeMAPS', verbose=True)
        
        if self.model_name == 'SVM':
            SVM_eGeMAPS = LinearSVM(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            SVM_eGeMAPS.run()
            y_pred_train, y_pred_dev = SVM_eGeMAPS.evaluate()
        elif self.model_name == 'RF':
            RF_eGeMAPS = RandomForest(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            RF_eGeMAPS.run()
            y_pred_train, y_pred_dev = RF_eGeMAPS.evaluate()
        
        get_UAR(y_pred_train, y_train, train_inst, self.model_name, self.feature_name, 'baseline', baseline=True, train_set=True, test=self.test)
        get_UAR(y_pred_dev, y_dev, dev_inst, self.model_name, self.feature_name, 'baseline', baseline=True, test=self.test)
        if not self.test:
            get_post_probability(y_pred_dev, y_dev, dev_inst, np.array([]), self.model_name, self.feature_name)

    def run_DeepSpectrum(self):
        """run classifier on DeepSpectrum feature (single modality)
        """
        print("\nbuilding a classifier on Deep features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('Deep', verbose=True)

        if self.model_name == 'SVM':
            SVM_Deep = LinearSVM(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            SVM_Deep.run()
            y_pred_train, y_pred_dev = SVM_Deep.evaluate()
        elif self.model_name == 'RF':
            RF_Deep = RandomForest(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            RF_Deep.run()
            y_pred_train, y_pred_dev = RF_Deep.evaluate()
        
        get_UAR(y_pred_train, y_train, train_inst, self.model_name, self.feature_name, 'baseline', baseline=True, train_set=True, test=self.test)
        get_UAR(y_pred_dev, y_dev, dev_inst, self.model_name, self.feature_name, 'baseline', baseline=True, test=self.test)
        if not self.test:
            get_post_probability(y_pred_dev, y_dev, dev_inst, np.array([]), self.model_name, self.feature_name)

    def run_BoAW(self):
        """run classifier on BoAW feature (single modality)
        """
        print("\nbuilding a classifier on BoAW features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('BoAW', verbose=True)

        if self.model_name == 'SVM':
            SVM_BoAW = LinearSVM(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            SVM_BoAW.run()
            y_pred_train, y_pred_dev = SVM_BoAW.evaluate()
        elif self.model_name == 'RF':
            RF_BoAW = RandomForest(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            RF_BoAW.run()
            y_pred_train, y_pred_dev = RF_BoAW.evaluate()
        
        get_UAR(y_pred_train, y_train, train_inst, self.model_name, self.feature_name, 'baseline', baseline=True, train_set=True, test=self.test)
        get_UAR(y_pred_dev, y_dev, dev_inst, self.model_name, self.feature_name, 'baseline', baseline=True, test=self.test)
        if not self.test:
            get_post_probability(y_pred_dev, y_dev, dev_inst, np.array([]), self.model_name, self.feature_name)

    def run_AU(self):
        """run classifier on AU feature (single modality)
        """
        print("\nbuilding a classifier on AU features (already session-level)")
        X_train, y_train, _, X_dev, y_dev, _ = load_proc_baseline_feature('AU', verbose=True)

        if self.model_name == 'SVM':
            SVM_AU = LinearSVM(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            SVM_AU.run()
            y_pred_train, y_pred_dev = SVM_AU.evaluate()
            session_prob = SVM_AU.get_session_probability()
        elif self.model_name == 'RF':
            RF_AU = RandomForest(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            RF_AU.run()
            y_pred_train, y_pred_dev = RF_AU.evaluate()
            session_prob = RF_AU.get_session_probability()
        
        get_UAR(y_pred_train, y_train, np.array([]), self.model_name, self.feature_name, 'baseline', baseline=True, train_set=True, test=self.test)
        get_UAR(y_pred_dev, y_dev, np.array([]), self.model_name, self.feature_name, 'baseline', baseline=True, test=self.test)
        if not self.test:
            get_post_probability(y_pred_dev, y_dev, np.array([]), session_prob, self.model_name, self.feature_name)

    def run_BoVW(self):
        """run classifier on BoVW feature (single modality)
        """
        print("\nbuilding a classifier on BoVW features (both frame-level and session-level)")
        X_train, y_train, train_inst, X_dev, y_dev, dev_inst = load_proc_baseline_feature('BoVW', verbose=True)

        if self.model_name == 'SVM':
            SVM_BoVW = LinearSVM(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            SVM_BoVW.run()
            y_pred_train, y_pred_dev = SVM_BoVW.evaluate()
        elif self.model_name == 'RF':
            RF_BoVW = RandomForest(self.feature_name, X_train, y_train, X_dev, y_dev, baseline=True, test=self.test)
            RF_BoVW.run()
            y_pred_train, y_pred_dev = RF_BoVW.evaluate()
        
        get_UAR(y_pred_train, y_train, train_inst, self.model_name, self.feature_name, 'baseline', baseline=True, train_set=True, test=self.test)
        get_UAR(y_pred_dev, y_dev, dev_inst, self.model_name, self.feature_name, 'baseline', baseline=True, test=self.test)
        if not self.test:
            get_post_probability(y_pred_dev, y_dev, dev_inst, np.array([]), self.model_name, self.feature_name)

    def run_fusion(self, feature_name_1, feature_name_2):
        """run late fusion on a pair of specified features
        """
        get_late_fusion_UAR(self.model_name, feature_name_1, feature_name_2, baseline=True)