import os
import json
import numpy as np
import pandas as pd
import keras_metrics as km
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import plot_model
from sklearn.metrics import classification_report, mean_squared_error


class SingleTaskDNN():
    def __init__(self, name, input_dim, num_class):
        self.fitted = False
        self.input_dim = input_dim
        self.num_class = num_class
        self.config = json.load(open('./config/model.json', 'r'))['singleDNN']
        self.model = None
        self.load_basics()
        self.name = '%s_hidden%.1f_batch%d_epoch%d_drop%.2f' % (name, self.hidden_ratio, self.batch_size, self.epochs, self.dropout)

    def load_basics(self):
        self.hidden_ratio = self.config['hidden_ratio']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.dropout = self.config['dropout']
        self.epochs = self.config['epochs']
        self.save_dir = self.config['save_dir']
        self.hidden_dim = [int(self.input_dim * self.hidden_ratio), 
                            int(self.input_dim / self.hidden_ratio),
                            int(self.input_dim / (self.hidden_ratio * 2))]
        self.output_dim = int(self.input_dim  / (self.hidden_ratio * 4))
        print("\nDNN classifier initialized and configuration loaded")

    def prepare_label(self, labels, dimension):
        print("\nprepare labels for DNN (classification)")
        multi_labels = np.zeros((len(labels), dimension), dtype=np.float64)
        for i in range(len(labels)):
            multi_labels[i][labels[i]-1] = 1.0
        return multi_labels

    def build_model(self):
        if not os.path.isdir(os.path.join(self.save_dir, self.name)):
            os.mkdir(os.path.join(self.save_dir, self.name))
            self.fitted = False
        else:
            self.fitted = True
        
        self.model = Sequential()
        self.model.add(Dense(self.hidden_dim[0], input_shape=(self.input_dim,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.hidden_dim[1]))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.hidden_dim[2]))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.num_class))
        self.model.add(Activation('sigmoid'))

        print(self.model.summary())

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[km.recall()])
        plot_model(self.model, show_shapes=True, to_file=os.path.join(self.save_dir, self.name, 'singleTaskDNN.png'))
    
    def train_model(self, X_train, y_train, X_dev, y_dev):
        if self.fitted:
            print("\nmodel already trained ---", self.name)
            self.load_model()
            return 
            
        y_train = self.prepare_label(y_train, self.num_class)
        y_dev = self.prepare_label(y_dev, self.num_class)

        csv_logger = CSVLogger(os.path.join(self.save_dir, self.name, "logger.csv"))
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{val_recall:.2f}.hdf5"), monitor=km.recall(), verbose=1, save_best_only=True, mode='max')
        callbacks_list = [csv_logger, checkpoint]

        self.model.fit(X_train, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    verbose=1,
                    validation_data=(X_dev, y_dev),
                    callbacks=callbacks_list)
        print("\nmodel trained and saved ---", self.name)
        self.save_model()
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test, batch_size=self.batch_size)
        y_pred = [np.argmax(y) + 1 for y in y_pred]
        print(classification_report(y_test, y_pred))

    def save_model(self):
        self.model.save_weights(os.path.join(self.save_dir, self.name, 'final_model.h5'))
        print("\nsaving completed ---", self.name)
    
    def load_model(self):
        self.model.load_weights(os.path.join(self.save_dir, self.name, 'final_model.h5'))
        print("\nloading completed ---", self.name)


class MultiTaskDNN(SingleTaskDNN):
    def __init__(self, name, input_dim, num_class):
        SingleTaskDNN.__init__(self, name, input_dim, num_class)
        self.config = json.load(open('./config/model.json', 'r'))['multiDNN']
        self.load_basics()
        self.name = '%s_hidden%.1f_batch%d_epoch%d_drop%.2f' % (name, self.hidden_ratio, self.batch_size, self.epochs, self.dropout)
    
    def prepare_regression_label(self, ymrs, inst):
        print("\nprepare labels for DNN (regression)")
        return np.array([ymrs[inst[i] - 1] for i in range(len(inst))])

    def build_model(self):
        if not os.path.isdir(os.path.join(self.save_dir, self.name)):
            os.mkdir(os.path.join(self.save_dir, self.name))
            self.fitted = False
        else:
            self.fitted = True
        
        input_layer = Input(shape=(self.input_dim, ))
        dense_layer = Dense(self.hidden_dim[0])(input_layer)
        dense_layer = Activation('relu')(dense_layer)
        dense_layer = Dropout(self.dropout)(dense_layer)
        dense_layer = Dense(self.hidden_dim[1])(dense_layer)
        dense_layer = Activation('relu')(dense_layer)
        dense_layer = Dropout(self.dropout)(dense_layer)
        dense_layer = Dense(self.hidden_dim[2])(dense_layer)
        dense_layer = Activation('relu')(dense_layer)
        dense_layer = Dense(self.output_dim)(dense_layer)
        dense_layer = Activation('relu')(dense_layer)

        # output layer for classification
        output_layer_c = Activation('softmax')(Dense(self.num_class)(dense_layer))
        # output layer for regression
        output_layer_r = Activation('linear')(Dense(1)(dense_layer))

        self.model = Model(inputs=input_layer, outputs=[output_layer_c, output_layer_r])
        print(self.model.summary())

        self.model.compile(loss=['binary_crossentropy', 'mean_squared_error'], optimizer='rmsprop', metrics=[km.recall()])
        plot_model(self.model, show_shapes=True, to_file=os.path.join(self.save_dir, self.name, 'multiTaskDNN.png'))
    
    def train_model(self, X_train, y_train_c, y_train_r, X_dev, y_dev_c, y_dev_r):
        # para y_<>_c: labels for classification
        # para y_<>_r: labels for regression
        if self.fitted:
            print("\nmodel already trained ---", self.name)
            self.load_model()
            return
        
        y_train_c = self.prepare_label(y_train_c, self.num_class)
        y_dev_c = self.prepare_label(y_dev_c, self.num_class)

        csv_logger = CSVLogger(os.path.join(self.save_dir, self.name, "logger.csv"))
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{val_recall:.2f}.hdf5"), monitor=km.recall(), verbose=1, save_best_only=True, mode='max')
        callbacks_list = [csv_logger, checkpoint]

        self.model.fit(X_train, [y_train_c, y_train_r],
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    verbose=1,
                    validation_data=(X_dev, [y_dev_c, y_dev_r]),
                    callbacks=callbacks_list)
        print("\nmodel trained and saved ---", self.name)
        self.save_model()
    
    def evaluate_model(self, X_train, y_train_c, y_train_r, X_dev, y_dev_c, y_dev_r):
        y_pred_train_c, y_pred_train_r = self.model.predict(X_train, batch_size=self.batch_size)
        y_pred_dev_c, y_pred_dev_r = self.model.predict(X_dev, batch_size=self.batch_size)
        y_pred_train_c = [np.argmax(y) + 1 for y in y_pred_train_c]
        y_pred_dev_c = [np.argmax(y) + 1 for y in y_pred_dev_c]

        print("--" * 20)
        print("performance on training set")
        print(classification_report(y_train_c, y_pred_train_c))
        print("mean squared error of regression")
        print(mean_squared_error(y_train_r, y_pred_train_r))
        
        print("--" * 20)
        print("performance on dev set")
        print(classification_report(y_dev_c, y_pred_dev_c))
        print("mean squared error of regression")
        print(mean_squared_error(y_dev_r, y_pred_dev_r))
