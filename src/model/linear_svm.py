import os
import json
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class LinearSVM():
    """
    Linear SVM classifier model
    ---
    Attributes
    -----------
    config: json.dict()
        configuration file for the model
    model_name: str
        model name "SVM"
    feature_name: str
        feature name given when initializing
    X_train, X_dev: np.array()
        training / development data given when initializing
    y_train, y_dev: np.array()
        training / development labels given when initializing
    parameters: dict()
        hyperparameters for the model
    model: svm.SVC()
        linear SVM instance in sklearn
    baseline: bool 
        whether baseline system or not
    test: bool
        whether test system or not
    -------------------------------------------------------------
    Functions
    -----------
    run(): public
        main function for the model
    train(): public
        train the model
    evaluate(): public
        evaluate the model
    tune(): public
        fine tune hyperparameters for the model
    get_session_probability(): public
        get posterior probability on session-level (FAU feature)
    """
    def __init__(self, feature_name, X_train, y_train, X_dev, y_dev, test=False, baseline=False):
        self.config = json.load(open('./config/model.json', 'r'))
        self.model_name = 'SVM'
        self.feature_name = feature_name
        # data normalization is a must for SVM
        self.X_train = preprocessing.normalize(X_train)
        self.X_dev = preprocessing.normalize(X_dev)
        self.y_train = y_train
        self.y_dev = y_dev
        self.parameters = dict()
        self.parameters['kernel'] = 'linear'
        self.parameters['C'] = None
        self.model = None
        self.baseline = baseline # indicate if baseline
        self.test = test # indicate if to test

    def run(self):
        """main function for the model
        """
        if self.test:
            self.parameters['C'] = 10.0

        if self.baseline:
            filename = os.path.join('config', 'baseline', '%s_%s_params.json' % (self.model_name, self.feature_name))
            if os.path.isfile(filename):
                self.parameters = json.load(open(filename, 'r'))

        if not self.parameters['C']:
            print("\nhyperparameters are not tuned yet")
            self.tune()
        
        # build SVM model
        if self.feature_name == 'AU':
            self.model = svm.SVC(kernel=self.parameters['kernel'],
                                C=self.parameters['C'],
                                probability=True)
        else:
            self.model = svm.SVC(kernel=self.parameters['kernel'],
                                C=self.parameters['C'])
        self.train()

    def train(self):
        """train the model
        """
        print("\ntraining a Linear SVM Classifier ...")
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate(self):
        """evaluate the model
        """
        print("\nevaluating the Linear SVM Classifier ...")
        y_pred_train = self.model.predict(self.X_train)
        y_pred_dev = self.model.predict(self.X_dev)

        print("\naccuracy on training set: %.3f" % metrics.accuracy_score(y_pred_train, self.y_train))
        print("\naccuracy on development set: %.3f" % metrics.accuracy_score(y_pred_dev, self.y_dev))
        return y_pred_train, y_pred_dev

    def tune(self):
        """fine tune hyperparameters for the model
        """
        import scipy.stats as stats
        parameters = {
            "kernel": ['linear'],
            "C": 10. ** np.arange(-5, 1)
        }
        print("\nrunning the Grid Search for Linear SVM classifier ...")
        clf = RandomizedSearchCV(svm.SVC(), 
                            parameters, 
                            cv=5, 
                            n_jobs=-1, 
                            verbose=3,
                            n_iter=30,
                            scoring='recall_macro',
                            pre_dispatch='2*n_jobs')
        
        clf.fit(self.X_train, self.y_train)
        print("\nfinal score for the tuned model\n", clf.score(self.X_train, self.y_train))
        print("\nbest hyperparameters for the tuned model\n", clf.best_params_)
        print("\ncross validation results (MEAN)\n", clf.cv_results_['mean_test_score'])
        print("\ncross validation results (STD)\n", clf.cv_results_['std_test_score'])

        self.parameters['C'] = clf.best_params_['C']

        if self.baseline:
            # write to model json file
            filename = os.path.join('config', 'baseline', '%s_%s_params.json' % (self.model_name, self.feature_name))
            with open(filename, 'w') as output:
                json.dump(clf.best_params_, output)
                output.write("\n")
            output.close()

    def get_session_probability(self):
        """get posterior probability on session-level (FAU feature)
        """
        return self.model.predict_proba(self.X_dev)