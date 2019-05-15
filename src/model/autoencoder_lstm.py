import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers import LSTM, RepeatVector, TimeDistributed
from keras.utils import plot_model

from src.utils.io import load_proc_baseline_feature
from src.model.autoencoder import AutoEncoder

# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))
model_config = json.load(open('./config/model.json', 'r'))
np.random.seed(1337)


class AutoEncoderLSTM():
    def __init__(self, test=False):
        self.X_train = None
        self.X_train_noisy = None
        self.X_dev = None
        self.X_dev_noisy = None
        self.dimension = [0] * 5
        self.hidden_ratio = None
        self.learning_rate = None
        self.epochs = None
        self.noise = None
        self.save_dir = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.model_config = json.load(open('./config/model.json', 'r'))
        self.load_basic(test=test)
        np.random.seed(1337)

    def load_basic(self, test):
        if not test:
            self.X_train, _, _, self.X_dev, _, _ = load_proc_baseline_feature('AU', verbose=True)
        else:
            # reshape input into [samples, timesteps, features]
            self.X_train = np.array(([[[x]*x for x in range(1, 10)]]))
            self.X_dev = np.array(([[[x+5]*x for x in range(1, 10)]]))
        
        print(self.X_train.shape)
        print(self.X_dev.shape)
        
        self.epochs = self.model_config['autoencoder-lstm']['epochs']

    def repeat(self, x):
        import keras.backend as K
        stepMatrix = K.ones_like(x[0][:,:,:1])
        latentMatrix = K.expand_dims(x[1], axis=1)
        return K.batch_dot(stepMatrix, latentMatrix)

    def build_model(self):
        self.autoencoder = Sequential()
        self.autoencoder.add(
            LSTM(200,
                activation='relu',
                input_shape=(None, 10))
        )
        # self.autoencoder.add(RepeatVector(9))
        self.autoencoder.add(Lambda(self.repeat)(Input(shape=(None, 10))))
        self.autoencoder.add(
            LSTM(200,
                activation='relu',
                return_sequences=True)
        )
        self.autoencoder.add(TimeDistributed(Dense(10)))
        self.autoencoder.compile(optimizer='adam', loss='mse')
        plot_model(self.autoencoder, show_shapes=True, to_file='./images/models/autoencoder_lstm.png')

    def train_model(self):
        self.autoencoder.fit(
            self.X_train, self.X_train,
            epochs=self.epochs,
            verbose=2
        )

    def evaluate_model(self):
        X_pred_train = self.autoencoder.predict(self.X_train, verbose=2)
        X_pred_dev = self.autoencoder.predict(self.X_dev, verbose=2)
        print(self.X_train, X_pred_train)
        print(self.X_dev, X_pred_dev)

    # def encode(self):
    
    # def decode(self):

    # def save_model(self):
    
    # def load_model(self):
