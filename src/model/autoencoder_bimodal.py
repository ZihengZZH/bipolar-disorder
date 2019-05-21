import os
import json
import numpy as np
from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.utils import plot_model
from src.model.autoencoder import AutoEncoder


class AutoEncoderBimodal(AutoEncoder):
    """
    """
    def __init__(self, name, X_train_A, X_train_V, X_dev_A, X_dev_V, sparse=False):
        # para name: name of bimodal AE
        # para X_train_A: pd.DataFrame
        # para X_train_V: pd.DataFrame
        # para X_dev_A: pd.DataFrame
        # para X_dev_V: pd.DataFrame
        self.X_train_A = X_train_A
        self.X_train_V = X_train_V
        self.X_dev_A = X_dev_A
        self.X_dev_V = X_dev_V
        AutoEncoder.__init__(self, name, X_train_A, X_dev_A, sparse=sparse)
        self.load_basic()
        self.dimension_A = self.X_train_A.shape[1]
        self.dimension_V = self.X_train_V.shape[1]

    def build_model(self):
        """build bimodal stacked deep autoencoder
        """
        hidden_dim = int((self.dimension_A + self.dimension_V) * self.hidden_ratio / 4)

        input_data_A = Input(shape=(self.dimension_A, ))
        input_data_V = Input(shape=(self.dimension_V, ))
        encoded_input = Input(shape=(hidden_dim, ))
        
        encoded_A = Dense(int(self.dimension_A * self.hidden_ratio), 
                        activation='relu')(input_data_A)
        encoded_V = Dense(int(self.dimension_V * self.hidden_ratio), 
                        activation='relu')(input_data_V)

        shared = Concatenate(axis=1)([encoded_A, encoded_V])
        if self.sparse:
            encoded = Dense(hidden_dim, 
                        activation='relu',
                        activity_regularizer=self._sparse_regularizer)(shared)
        else:
            encoded = Dense(hidden_dim, activation='relu')(shared)
        
        decoded_A = Dense(int(self.dimension_A * self.hidden_ratio), 
                        activation='relu')(encoded)
        decoded_V = Dense(int(self.dimension_V * self.hidden_ratio), 
                        activation='relu')(encoded)

        decoded_A = Dense(self.dimension_A, activation='sigmoid')(decoded_A)
        decoded_V = Dense(self.dimension_V, activation='sigmoid')(decoded_V)

        self.autoencoder = Model(inputs=[input_data_A, input_data_V], outputs=[decoded_A, decoded_V])
        self.encoder = Model([input_data_A, input_data_V], encoded)
        # only for visual
        self.decoder = Model(encoded_input, decoded_V(encoded_input)) 

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
        """train bimodal stacked deep autoencoder
        """
        self.autoencoder.fit([self.X_train_A, self.X_train_V],
                            [self.X_train_A, self.X_train_V],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            validation_data=(
                                [self.X_dev_A, self.X_dev_V],
                                [self.X_dev_A, self.X_dev_V]
                            ))
        self.save_model()

    def encode(self, X_1_A, X_1_V, X_2_A, X_2_V):
        """encode bimodal input to latent representation
        """
        encoded_train = self.encoder.predict([X_1_A, X_1_V])
        encoded_dev = self.encoder.predict([X_2_A, X_2_V])
        self.save_representation(encoded_train, encoded_dev)
    
    def decode(self, encoded_pre):
        """decode latent representation to bimodal input
        """
        decoded_input_V = self.decoder.predict(encoded_pre)
        return decoded_input_V