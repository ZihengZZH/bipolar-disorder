import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.utils import plot_model
from src.model.autoencoder import AutoEncoder


class AutoEncoderBimodal(AutoEncoder):
    def __init__(self, X_train_A, X_train_V, X_dev_A, X_dev_V):
        self.X_train_A = X_train_A
        self.X_train_V = X_train_V
        self.X_dev_A = X_dev_A
        self.X_dev_V = X_dev_V
        AutoEncoder.__init__(self, 'bimodal', X_train_A, X_dev_A)
        self.load_basic()
    
    def build_model(self):
        self.dimension[2] = int(self.dimension[2] * 2)

        input_data_A = Input(shape=(self.dimension[0], ))
        input_data_V = Input(shape=(self.dimension[0], ))
        
        encoded_A = Dense(self.dimension[1], activation='relu')(input_data_A)
        encoded_V = Dense(self.dimension[1], activation='relu')(input_data_V)

        shared = Concatenate(axis=1)([encoded_A, encoded_V])
        encoded = Dense(self.dimension[2], activation='relu')(shared)

        decoded_A = Dense(self.dimension[3], activation='relu')(encoded)
        decoded_V = Dense(self.dimension[3], activation='relu')(encoded)

        decoded_A = Dense(self.dimension[4], activation='sigmoid')(decoded_A)
        decoded_V = Dense(self.dimension[4], activation='sigmoid')(decoded_V)

        self.autoencoder = Model(input=[input_data_A, input_data_V], outputs=[decoded_A, decoded_V])

        self.encoder = Model([input_data_A, input_data_V], encoded)

        # configure model
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        print("--" * 20)
        print("autoencoder")
        print(self.autoencoder.summary())
        print("--" * 20)
        print("encoder")
        print(self.encoder.summary())
        plot_model(self.autoencoder, show_shapes=True, to_file='./images/models/autoencoder_bimodal.png')

    def train_model(self):
        self.autoencoder.fit([self.X_train_A, self.X_train_V],
                            [self.X_train_A, self.X_train_V],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            validation_data=(
                                [self.X_dev_A, self.X_dev_V],
                                [self.X_dev_A, self.X_dev_V]))
        self.save_model()