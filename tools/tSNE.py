"""
t-Distributed Stochastic Neighbouring Entities
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tSNE_doc2vec(dm=True):
    model_path = '/media/zzh/Ziheng-700G/Dataset/bipolar-disorder/pre-trained/doc2vec/Doc2Vec(dm-m,d50,n5,w10,mc2,t8)' if dm else '/media/zzh/Ziheng-700G/Dataset/bipolar-disorder/pre-trained/doc2vec/Doc2Vec(dbow,d50,n5,mc2,t8)'

    if os.path.isfile(os.path.join(model_path, 'evaluation.txt')):
        X_3d = []
        json_data = json.load(open(os.path.join(model_path, 'state.txt'), 'r'))
        X = json_data[0]['projections']
        for i in range(len(X)):
            X_3d.append([X[i]['tsne-0'], X[i]['tsne-1'], X[i]['tsne-2']])
        X_3d = np.array(X_3d)

        y = pd.read_csv(os.path.join(model_path, 'label.tsv'), sep='\t')
        y_name = y.iloc[:,0].tolist()
        y_label = y.iloc[:,1].tolist()

        assert len(X_3d) == len(y_name) == len(y_label)

        X_3d_1 = np.array([X_3d[i] for i in range(len(X_3d)) if y_label[i] == 1])
        X_3d_2 = np.array([X_3d[i] for i in range(len(X_3d)) if y_label[i] == 2])
        X_3d_3 = np.array([X_3d[i] for i in range(len(X_3d)) if y_label[i] == 3])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_3d_1[:,0], X_3d_1[:,1], X_3d_1[:,2], marker='o', c='r', label='depression')
        ax.scatter(X_3d_2[:,0], X_3d_2[:,1], X_3d_2[:,2], marker='v', c='g', label='hypo-mania')
        ax.scatter(X_3d_3[:,0], X_3d_3[:,1], X_3d_3[:,2], marker='s', c='b', label='mania')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tSNE_doc2vec(dm=True)