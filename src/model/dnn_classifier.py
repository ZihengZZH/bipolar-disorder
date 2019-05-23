import os
import json
import numpy as np
import pandas as pd
import keras_metrics as km
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
from sklearn.metrics import classification_report


class SingleTaskDNN():
    def __init__(self, name, input_dim):
        self.name = name
        self.input_dim = input_dim
        self.config = json.load(open('./config/model.json', 'r'))['singleDNN']
        self.model = None
        self.hidden_dim = None
        self.hidden_ratio = None
        self.learning_rate = None
        self.batch_size = None
        self.dropout = None
        self.epochs = None
        self.load_basics()

    def load_basics(self):
        self.hidden_ratio = self.config['hidden_ratio']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.dropout = self.config['dropout']
        self.epochs = self.config['epochs']
        self.hidden_dim = int(self.input_dim * self.hidden_ratio)
        self.output_dim = int(self.input_dim / self.hidden_ratio)

    def prepare_label(self, labels):
        multi_labels = np.zeros((len(labels), max(labels)), dtype=np.float64)
        for i in range(len(labels)):
            multi_labels[i][labels[i]-1] = 1.0
        return multi_labels

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_dim, input_shape=(self.input_dim,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('relu'))
        self.model.add(Dense(3))
        self.model.add(Activation('sigmoid'))

        print(self.model.summary())

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[km.recall()])
        plot_model(self.model, show_shapes=True, to_file='./images/models/singleTaskDNN.png')
    
    def train_model(self, X_train, y_train, X_dev, y_dev):
        y_train = self.prepare_label(y_train)
        y_dev = self.prepare_label(y_dev)

        self.model.fit(X_train, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    verbose=1,
                    validation_data=(X_dev, y_dev))
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test, batch_size=self.batch_size)
        y_pred = [np.argmax(y) + 1 for y in y_pred]
        print(classification_report(y_test, y_pred))


class MultiTaskDNN(SingleTaskDNN):
    def __init__(self):
        pass

    def build_model(self):
        pass
    
    def train_model(self, X_train, y_train_c, y_train_r, X_dev, y_dev_c, y_dev_r):
        # para y_train_c: training labels (classification)
        # para y_train_r: training labels (regression)
        pass
    
    def evaluate_model(self):
        pass

    def save_model(self):
        pass
    
    def load_model(self):
        pass