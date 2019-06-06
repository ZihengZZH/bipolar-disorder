import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_metrics as km
import keras.backend as K
from itertools import product
from functools import partial
from keras.metrics import categorical_accuracy
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.preprocessing import minmax_scale
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
        self.hidden_dim = [int(self.input_dim / self.hidden_ratio), 
                            int(self.input_dim / (self.hidden_ratio * 2)),
                            int(self.input_dim / (self.hidden_ratio * 4))]
        self.output_dim = int(self.input_dim  / (self.hidden_ratio * 8))
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
    
        def w_categorical_crossentropy(y_true, y_pred, weights):
            nb_cl = len(weights)
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1)
            y_pred_max = K.expand_dims(y_pred_max, 1)
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
            return K.categorical_crossentropy(y_pred, y_true) * final_mask

        # weight to be customized
        w = np.ones((3,3))

        ncce = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w)
        ncce.__name__ ='w_categorical_crossentropy'

        self.model.compile(loss=ncce, optimizer='sgd', metrics=[km.recall()])
        plot_model(self.model, show_shapes=True, to_file=os.path.join(self.save_dir, self.name, 'singleTaskDNN.png'))
    
    def train_model(self, X_train, y_train, X_dev, y_dev):
        if self.fitted:
            print("\nmodel already trained ---", self.name)
            self.load_model()
            return 
            
        y_train = self.prepare_label(y_train, self.num_class)
        y_dev = self.prepare_label(y_dev, self.num_class)

        csv_logger = CSVLogger(os.path.join(self.save_dir, self.name, "logger.csv"))
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{val_loss:04f}.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='max')
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
        output_layer_c = Dense(self.num_class, activation='softmax', name='output_c')(dense_layer)
        # output layer for regression
        output_layer_r = Dense(1, activation='linear', name='output_r')(dense_layer)

        self.model = Model(inputs=input_layer, outputs=[output_layer_c, output_layer_r])
        print(self.model.summary())

        self.model.compile(loss={'output_c':'categorical_crossentropy', 
                                'output_r':'mean_squared_error'}, 
                            optimizer='adam',
                            loss_weights={'output_c': 1.0,
                                        'output_r': 0.1}, 
                            metrics={'output_c': km.recall(),
                                    'output_r': 'mse'})
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
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{val_loss:04f}.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [csv_logger, checkpoint]

        self.model.fit(X_train, {'output_c': y_train_c, 
                                'output_r': y_train_r},
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    verbose=1,
                    validation_data=(X_dev, [y_dev_c, y_dev_r]),
                    callbacks=callbacks_list)
        print("\nmodel trained and saved ---", self.name)
        self.save_model()
    
    def evaluate_model(self, X_train, y_train_c, y_train_r, X_dev, y_dev_c, y_dev_r, verbose=False):
        y_pred_train_c, y_pred_train_r = self.model.predict(X_train, batch_size=self.batch_size)
        y_pred_dev_c, y_pred_dev_r = self.model.predict(X_dev, batch_size=self.batch_size)
        
        y_pred_train = [np.argmax(y) + 1 for y in y_pred_train_c]
        y_pred_dev = [np.argmax(y) + 1 for y in y_pred_dev_c]

        assert len(y_train_c) == len(y_pred_train)
        assert len(y_dev_c) == len(y_pred_dev)

        if verbose:
            print("\ntraining parition")
            for i in range(len(y_pred_train_c)):
                print(y_pred_train_c[i], y_pred_train[i], y_train_c[i])
            print("\ndevelopment parition")
            for j in range(len(y_pred_dev_c)):
                print(y_pred_dev_c[j], y_pred_dev[j], y_dev_c[j])

        print("--" * 20)
        print("performance on training set")
        print(classification_report(y_train_c, y_pred_train))
        print("mean squared error of regression")
        print(mean_squared_error(y_train_r, y_pred_train_r))
        
        print("--" * 20)
        print("performance on dev set")
        print(classification_report(y_dev_c, y_pred_dev))
        print("mean squared error of regression")
        print(mean_squared_error(y_dev_r, y_pred_dev_r))
