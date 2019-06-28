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
from keras.layers import Input, Dense, Dropout, Activation, Layer
from keras.initializers import Constant
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report, mean_squared_error


class SingleTaskDNN():
    """
    Single-Task Deep Neural Nets
    ---
    Attributes
    -----------
    name: str
        model name
    input_dim: int
        dimension of input data (fused feature)
    num_class: int
        3 classes: mania / hypomania / depressive
    model: keras.models.Model
        keras Model as Deep Neural Net
    ---------------------------------------
    Functions
    -----------
    load_basics(): public
        load basic configuration
    prepare_label(): public
        transform 1-size 3-class labels to fit model
    build_model(): public
        build & compile the keras model
    train_model(): public
        train the model with provided data
    evaluate_model(): public
        evaluate the model on training/dev sets
    save_model(): public
        save model to external files
    load_model(): public
        load model from external files
    """
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
        self.model.add(Dense(self.hidden_dim[1]))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(self.dropout))
        # self.model.add(Dense(self.hidden_dim[2]))
        # self.model.add(Activation('relu'))
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

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])
        plot_model(self.model, show_shapes=True, to_file=os.path.join(self.save_dir, self.name, 'singleTaskDNN.png'))
    
    def train_model(self, X_train, y_train, X_dev, y_dev):
        if self.fitted:
            print("\nmodel already trained ---", self.name)
            self.load_model()
            return 
            
        y_train = self.prepare_label(y_train, self.num_class)
        y_dev = self.prepare_label(y_dev, self.num_class)

        csv_logger = CSVLogger(os.path.join(self.save_dir, self.name, "logger.csv"))
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{val_loss:04f}.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor='val_acc', mode='max')
        callbacks_list = [csv_logger, checkpoint, early]

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
    """
    Multi-Task Deep Neural Nets (built on SingleTaskDNN)
    ---
    Attributes
    -----------
    name: str
        model name
    input_dim: int
        dimension of input data (fused feature)
    num_class: int
        3 classes: mania / hypomania / depressive
    model: keras.models.Model
        keras Model as Deep Neural Net
    ---------------------------------------
    Functions
    -----------
    prepare_regression_label(): public
        transform 1-size 3-class labels to fit model
    build_model(): public
        build & compile the keras model
    train_model(): public
        train the model with provided data
    evaluate_model(): public
        evaluate the model on training/dev sets
    """
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
        dense_layer = Dense(self.hidden_dim[1])(dense_layer)
        dense_layer = Activation('relu')(dense_layer)
        # dense_layer = Dropout(self.dropout)(dense_layer)
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
                                        'output_r': 0.2}, 
                            metrics={'output_c': 'accuracy',
                                    'output_r': 'mean_squared_error'})
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
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{val_loss:04f}.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [csv_logger, checkpoint]

        self.model.fit(X_train, {'output_c': y_train_c, 
                                'output_r': y_train_r},
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    verbose=1,
                    validation_data=(X_dev, {'output_c': y_dev_c, 
                                            'output_r': y_dev_r}),
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


class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
            initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

class MultiLossDNN(SingleTaskDNN):
    def __init__(self, name, input_dim, num_class):
        SingleTaskDNN.__init__(self, name, input_dim, num_class)
        self.config = json.load(open('./config/model.json', 'r'))['multiDNN']
        self.load_basics()
        self.name = '%s_hidden%.1f_batch%d_epoch%d_drop%.2f_multiloss' % (name, self.hidden_ratio, self.batch_size, self.epochs, self.dropout)

    def run(self, X_train, y_train_c, y_train_r, X_dev, y_dev_c, y_dev_r):
        if not os.path.isdir(os.path.join(self.save_dir, self.name)):
            os.mkdir(os.path.join(self.save_dir, self.name))
            self.fitted = False
        else:
            self.fitted = True
        
        y_train_c = self.prepare_label(y_train_c, self.num_class)
        y_dev_c = self.prepare_label(y_dev_c, self.num_class)

        input_layer = Input(shape=(self.input_dim, ))
        dense_layer = Dense(self.hidden_dim[1])(input_layer)
        dense_layer = Activation('relu')(dense_layer)

        # output layer for classification
        output_layer_c = Dense(3, activation='sigmoid', name='output_c')(dense_layer)
        # output layer for regression
        output_layer_r = Dense(1, activation='linear', name='output_r')(dense_layer)

        true_layer_c = Input(shape=(3, ), name='true_c')
        true_layer_r = Input(shape=(1, ), name='true_r')
        out = CustomMultiLossLayer(nb_outputs=2)([true_layer_c, true_layer_r, output_layer_c, output_layer_r])

        self.prediction_model = Model(input=input_layer, outputs=[output_layer_c, output_layer_r])
        self.trainable_model = Model(inputs=[input_layer, true_layer_c, true_layer_r], outputs=out)

        print(self.prediction_model.summary())
        print(self.trainable_model.summary())

        self.trainable_model.compile(loss=None, optimizer='adam')
        plot_model(self.trainable_model, show_shapes=True, to_file=os.path.join(self.save_dir, self.name, 'multiTaskDNN.png'))
        hist = self.trainable_model.fit(X_train, y_train_c, y_train_r,
                                        nb_epoch=self.epochs,
                                        verbose=1,
                                        validation_data=(X_dev, y_dev_c, y_dev_r))
        
        import pylab
        pylab.plot(hist.history['loss'])
        pylab.plot(hist.history['val_loss'])