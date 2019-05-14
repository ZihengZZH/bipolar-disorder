import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.utils import plot_model

from src.utils.io import load_proc_baseline_feature


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))
model_config = json.load(open('./config/model.json', 'r'))
np.random.seed(1337)


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
    def __init__(self, feature_name):
        self.name = feature_name
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
        self.load_basic()
    
    def load_basic(self):
        self.X_train, _, _, self.X_dev, _, _ = load_proc_baseline_feature(self.name, verbose=True)

        self.hidden_ratio = model_config['autoencoder']['hidden_ratio']
        self.learning_rate = model_config['autoencoder']['learning_rate']
        self.batch_size = model_config['autoencoder']['batch_size']
        self.epochs = model_config['autoencoder']['epochs']
        self.noise = model_config['autoencoder']['noise']
        self.save_dir = model_config['autoencoder']['save_dir']
        self.dimension[0], _ = self.X_train.shape
        self.dimension[1] = int(self.dimension[0] * self.hidden_ratio)
        self.dimension[2] = int(self.dimension[1] * self.hidden_ratio)
        self.dimension[3] = self.dimension[1]
        self.dimension[4] = self.dimension[0]
        # prepare noisy data
        self.X_train_noisy = self.X_train + np.random.normal(loc=0.5, scale=0.5, size=self.X_train.shape)
        self.X_dev_noisy = self.X_dev + np.random.normal(loc=0.5, scale=0.5, size=self.X_dev.shape)

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
        print(self.autoencoder.summary())
        print(self.encoder.summary())
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
        """encode raw data to hidden representation
        """
        encoded_repre = self.encoder.predict(self.X_dev)
        return encoded_repre
    
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