import os
import json
import numpy as np
import pandas as pd
import keras_metrics as km
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
from sklearn.metrics import classification_report


class SingleTaskDNN():
    def __init__(self, name, input_dim, num_class):
        self.name = name
        self.input_dim = input_dim
        self.num_class = num_class
        self.config = json.load(open('./config/model.json', 'r'))['singleDNN']
        self.model = None
        self.load_basics()

    def load_basics(self):
        self.hidden_ratio = self.config['hidden_ratio']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.dropout = self.config['dropout']
        self.epochs = self.config['epochs']
        self.hidden_dim = int(self.input_dim * self.hidden_ratio)
        self.output_dim = [int(self.input_dim / self.hidden_ratio),
                            int(self.input_dim  / (self.hidden_ratio * 2))]

    def prepare_label(self, labels, dimension):
        multi_labels = np.zeros((len(labels), dimension), dtype=np.float64)
        for i in range(len(labels)):
            multi_labels[i][labels[i]-1] = 1.0
        return multi_labels

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_dim, input_shape=(self.input_dim,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.output_dim[0]))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.output_dim[1]))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.num_class))
        self.model.add(Activation('sigmoid'))

        print(self.model.summary())

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[km.recall()])
        plot_model(self.model, show_shapes=True, to_file='./images/models/singleTaskDNN.png')
    
    def train_model(self, X_train, y_train, X_dev, y_dev):
        y_train = self.prepare_label(y_train, self.num_class)
        y_dev = self.prepare_label(y_dev, self.num_class)

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
    def __init__(self, name, input_dim, num_class, reg_range):
        self.reg_range = reg_range
        self.config = json.load(open('./config/model.json', 'r'))['multiDNN']
        SingleTaskDNN.__init__(self, name, input_dim, num_class)
        self.load_basics()
    
    def prepare_regression_label(self, ymrs, inst):
        return [ymrs[inst[i] - 1] for i in range(len(inst))] 

    def build_model(self):
        input_layer = Input(shape=(self.input_dim, ))
        dense_layer = Dense(self.hidden_dim)(input_layer)
        dense_layer = Activation('relu')(dense_layer)
        dense_layer = Dropout(self.dropout)(dense_layer)
        dense_layer = Dense(self.output_dim[0])(dense_layer)
        dense_layer = Activation('relu')(dense_layer)
        dense_layer = Dropout(self.dropout)(dense_layer)
        dense_layer = Dense(self.output_dim[1])(dense_layer)
        dense_layer = Activation('relu')(dense_layer)

        # output layer for classification
        output_layer_c = Activation('sigmoid')(Dense(self.num_class)(dense_layer))
        # output layer for regression
        output_layer_r = Activation('sigmoid')(Dense(self.reg_range)(dense_layer))

        self.model = Model(inputs=input_layer, outputs=[output_layer_c, output_layer_r])
        print(self.model.summary())

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[km.recall()])
        plot_model(self.model, show_shapes=True, to_file='./images/models/multiTaskDNN.png')
    
    def train_model(self, X_train, y_train_c, y_train_r, X_dev, y_dev_c, y_dev_r):
        # para y_<>_c: labels for classification
        # para y_<>_r: labels for regression
        y_train_c = self.prepare_label(y_train_c, self.num_class)
        y_train_r = self.prepare_label(y_train_r, self.reg_range)
        y_dev_c = self.prepare_label(y_dev_c, self.num_class)
        y_dev_r = self.prepare_label(y_dev_r, self.reg_range)

        self.model.fit(X_train, [y_train_c, y_train_r],
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    verbose=1,
                    validation_data=(X_dev, [y_dev_c, y_dev_r]))
    
    def evaluate_model(self, X_test, y_test_c, y_test_r):
        y_pred_c, y_pred_r = self.model.predict(X_test, batch_size=self.batch_size)
        y_pred_c = [np.argmax(y) + 1 for y in y_pred_c]
        y_pred_r = [np.argmax(y) + 1 for y in y_pred_r]
        print(classification_report(y_test_c, y_pred_c))
        print(classification_report(y_test_r, y_pred_r))
