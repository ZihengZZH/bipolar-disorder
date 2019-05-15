import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import plot_model


class AutoEncoder():
    """
    Stacked Denoising Autoencoder (SDAC) to encode visual data
    ---
    Attributes
    -----------
    name: str
    X_train, X_dev: pd.DataFrame
    ---------------------------------------
    Functions
    -----------
    build_model(): public

    """
    def __init__(self, name, X_train, X_dev, noisy=True):
        self.name = name
        self.X_train = X_train
        self.X_train_noisy = None
        self.X_dev = X_dev
        self.X_dev_noisy = None
        self.noisy = noisy
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
        self.load_basic()
        np.random.seed(1337)
    
    def load_basic(self):
        self.hidden_ratio = self.model_config['autoencoder']['hidden_ratio']
        self.learning_rate = self.model_config['autoencoder']['learning_rate']
        self.batch_size = self.model_config['autoencoder']['batch_size']
        self.epochs = self.model_config['autoencoder']['epochs']
        self.noise = self.model_config['autoencoder']['noise']
        self.save_dir = self.model_config['autoencoder']['save_dir']
        self.dimension[0] = self.X_train.shape[1]
        self.dimension[1] = int(self.dimension[0] * self.hidden_ratio)
        self.dimension[2] = int(self.dimension[1] * self.hidden_ratio)
        self.dimension[3] = self.dimension[1]
        self.dimension[4] = self.dimension[0]

        if self.noisy:
            # prepare noisy data
            self.X_train_noisy = self.X_train + np.random.normal(loc=0.5, scale=0.5, size=self.X_train.shape)
            self.X_dev_noisy = self.X_dev + np.random.normal(loc=0.5, scale=0.5, size=self.X_dev.shape)
        else:
            self.X_train_noisy = self.X_train
            self.X_dev_noisy = self.X_dev

        assert self.X_train_noisy.shape == self.X_train.shape
        assert self.X_dev_noisy.shape == self.X_dev.shape

    def build_model(self):
        """build stacked denoising autoencoder model
        """
        # input placeholder
        input_data = Input(shape=(self.dimension[0], ))

        # encoder part
        encoded = Dense(self.dimension[1], activation='relu')(input_data)
        encoded = Dense(self.dimension[2], activation='relu')(encoded)
        # decoder part
        decoded = Dense(self.dimension[3], activation='relu')(encoded)
        decoded = Dense(self.dimension[4], activation='sigmoid')(decoded)

        # maps input to its reconstruction
        self.autoencoder = Model(input_data, decoded)
        # maps input to its representation
        self.encoder = Model(input_data, encoded)

        # encoded input placeholder
        encoded_input = Input(shape=(self.dimension[2], ))
        # retrieve layers of autoencoder
        decoder_1 = self.autoencoder.layers[-2]
        decoder_2 = self.autoencoder.layers[-1]
        # maps representation to input
        self.decoder = Model(encoded_input, decoder_2(decoder_1(encoded_input)))

        # configure model
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        print("--" * 20)
        print("autoencoder")
        print(self.autoencoder.summary())
        print("--" * 20)
        print("encoder")
        print(self.encoder.summary())
        print("--" * 20)
        print("decoder")
        print(self.decoder.summary())

    def train_model(self):
        """train stacked denoising autoencoder model
        """
        self.autoencoder.fit(self.X_train_noisy, self.X_train, 
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            validation_data=(self.X_dev_noisy, self.X_dev))
        self.save_model()
    
    def encode(self):
        """encode raw input to latent representation
        """
        encoded_pre = self.encoder.predict(self.X_dev)
        return encoded_pre

    def decode(self, encoded_pre):
        """decode latent representation to raw input
        """
        decoded_input = self.decoder.predict(encoded_pre)
        return decoded_input
    
    def save_model(self):
        """save stacked denoising autoencoder model to external file
        """
        filename = '%s-h%.2f-l%.2f-b%d-e%d-n%.1f-%s.h5' % (
            self.name, self.hidden_ratio, self.learning_rate,
            self.batch_size, self.epochs, self.noise, 
            datetime.datetime.now().strftime('%d%m-%H%M')
        )
        self.autoencoder.save_weights(os.path.join(self.save_dir, filename))

    def load_model(self):
        """load stacked denoising autoencoder model from external file
        """
        weights_list = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]
        print("Full list of pre-trained models")
        print("--"*20)
        for idx, name in enumerate(weights_list):
            print("no.%d model with name %s" % (idx, name))
        print("--"*20)
        choose_model = None
        for _ in range(3):
            try:
                choose_model = input("\nPlease make your choice\t")
                weights_name = weights_list[int(choose_model)]
                self.autoencoder.load_weights(os.path.join(self.save_dir, weights_name))
                break
            except:
                print("\nWrong input! Please start over")

    def vis_model(self):
        pass