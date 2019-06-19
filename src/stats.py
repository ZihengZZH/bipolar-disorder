import os
import math
import itertools
import numpy as np
from smart_open import smart_open
from progressbar import ProgressBar
from src.model.random_forest import RandomForest


class PermutationTest():
    def __init__(self, model_index_1, model_index_2, R=5000, baseline=False):
        self.model_path = smart_open('./pre-trained/DDAE/model_list.txt', 'rb', encoding='utf-8')
        self.model_list = []
        self.kernel_list = []
        self.R = R
        self.baseline = baseline

        for _, line_AV in enumerate(self.model_path):
            line_AV = str(line_AV).replace('\n', '')
            self.model_list.append(line_AV[:-2])
            self.kernel_list.append(line_AV[-2:])
        
        self.model_1 = self.model_list[model_index_1]
        self.model_2 = self.model_list[model_index_2]
        self.kernel_1 = self.kernel_list[model_index_1]
        self.kernel_2 = self.kernel_list[model_index_2]

        print(self.model_1[19:])
        print(self.model_2[19:])
    
    def run(self):
        y_pred_1 = self.get_prediction_res(self.model_1, self.kernel_1)
        
        if self.baseline:
            self.model_2 = 'baseline_eGeMAPS'
            y_pred_2 = np.load(os.path.join('pre-trained', 'baseline', 'RF_eGeMAPS_results.npy'))
        else:
            y_pred_2 = self.get_prediction_res(self.model_2, self.kernel_2)
        
        print("PREDICTION BY MODEL 1 %s\n" % self.model_1, y_pred_1)
        print("PREDICTION BY MODEL 2 %s\n" % self.model_2, y_pred_2)
        
        p_value = self.run_permutation_test(y_pred_1, y_pred_2, R=self.R)
        print("p-value for permutation test", p_value)
        self.save_result(p_value)

    def get_prediction_res(self, model_path, kernel):
        X_train = np.load(os.path.join(model_path, 'X_train_tree_%d.npy' % int(kernel)))
        X_dev = np.load(os.path.join(model_path, 'X_dev_tree_%d.npy' % int(kernel)))
        y_train = np.load(os.path.join(model_path, 'label_train.npy'))
        y_dev = np.load(os.path.join(model_path, 'label_dev.npy'))
        random_forest = RandomForest('permutation_test', X_train, y_train, X_dev, y_dev, test=True)
        random_forest.run()
        _, y_pred = random_forest.evaluate()
        return y_pred

    def run_permutation_test(self, result_A, result_B, R=0):
        """run Monte Carlo Permutation test
        """
        # para result_A: np.array
        # para result_B: np.array
        # para R: preset number of permuted samples
        # para R: if R != 0, called Monte Carlo Permutation test
        p_value, no_larger = .0, 0
        assert len(result_A) == len(result_B), "ERROR! LENGTHS MISMATCH"

        n, k = len(result_A), 0
        diff = np.abs(np.mean(result_A) - np.mean(result_B))
        results = np.concatenate([result_A, result_B])
        bar = ProgressBar()
        for _ in bar(range(R)):
            np.random.shuffle(results)
            k += diff <= np.abs(np.mean(results[:n]) - np.mean(results[n:]))
        return round(((k+1) / (R+1)), 5)

    def save_result(self, p_value):
        if self.baseline:
            filename = os.path.join('results', 'stats', '%s_%s_stats_results.txt' % (self.model_1[19:], self.model_2))
        else:
            filename = os.path.join('results', 'stats', '%s_%s_stats_results.txt' % (self.model_1[19:], self.model_2[19:]))
        with smart_open(filename, 'wb', encoding='utf-8') as f:
            f.write("%s\n" % self.model_1[19:])
            if self.baseline:
                f.write("%s\n" % self.model_2)
            else:
                f.write("%s\n" % self.model_2[19:])
            f.write("%f\n" % p_value)