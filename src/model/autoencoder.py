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
        model name
    X_train, X_dev: pd.DataFrame
        original data without noise
    X_train_noisy, X_dev_noisy: pd.DataFrame
        noisy data generated with Gaussian noise
    noisy: bool
        whether or not to involve denoising fashion
    autoencoder: keras.models.Model
        keras Model mapping input to reconstructed input
    encoder: keras.models.Model
        keras Model mapping input to latent representation
    decoder: keras.models.Model
        keras Model mapping latent representation to input
    dimension: list
    hidden_ratio: float
    learning_rate: float
    epochs: int
    noise: float
    save_dir: str
    model_config: json.load()
    ---------------------------------------
    Functions
    -----------
    load_basic(): public
        load basic data and configuration for model
    build_model(): public
        build stacked denoising autoencoder model
    train_model(): public
        train stacked denoising autoencoder model
    encode(): public
        encode raw input to latent representation
    decode(): public
        decode latent representation to raw input
    save_model(): public
        save stacked denoising autoencoder model to external file
    load_model(): public
        load stacked denoising autoencoder model from external file
    """
    def __init__(self, name, X_train, X_dev, noisy=True):
        # para name: str
        # para X_train: pd.DataFrame
        # para X_dev: pd.DataFrame
        self.name = name
        self.X_train = X_train
        self.X_train_noisy = None
        self.X_dev = X_dev
        self.X_dev_noisy = None
        self.noisy = noisy
        # AE model
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

        self.dimension = [0] * 5
        self.hidden_ratio = None
        self.learning_rate = None
        self.epochs = None
        self.noise = None
        self.save_dir = None
        self.model_config = json.load(open('./config/model.json', 'r'))['autoencoder']
        self.load_basic()
        np.random.seed(1337)

    def load_basic(self):
        """load basic data and configuration for model
        """
        self.hidden_ratio = self.model_config['hidden_ratio']
        self.learning_rate = self.model_config['learning_rate']
        self.batch_size = self.model_config['batch_size']
        self.epochs = self.model_config['epochs']
        self.noise = self.model_config['noise']
        self.save_dir = self.model_config['save_dir']
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
    
    def encode(self, X_1, X_2):
        """encode raw input to latent representation
        """
        encoded_train = self.encoder.predict(X_1)
        encoded_dev = self.encoder.predict(X_2)
        self.save_representation(encoded_train, encoded_dev)

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
        print("\nfull list of pre-trained models")
        print("--"*20)
        for idx, name in enumerate(weights_list):
            print("no.%d model with name %s" % (idx, name))
        print("--"*20)
        choose_model = None
        for _ in range(3):
            try:
                choose_model = input("\nmake your choice: ")
                weights_name = weights_list[int(choose_model)]
                self.autoencoder.load_weights(os.path.join(self.save_dir, weights_name))
                break
            except ValueError:
                print("\nWrong input! Please start over")

    def save_representation(self, encoded_train, encoded_dev):
        """save encoded representation to external file
        """
        encoded_dir = self.model_config['encoded_dir']
        np.save(os.path.join(encoded_dir, 'encoded_train_%s' % self.name), encoded_train)
        np.save(os.path.join(encoded_dir, 'encoded_dev_%s' % self.name), encoded_dev)
    
    def load_presentation(self):
        """load encoded representation from external file
        """
        encoded_dir = self.model_config['encoded_dir']
        encoded_train = np.load(os.path.join(encoded_dir, 'encoded_train_%s.npy' % self.name))
        encoded_dev = np.load(os.path.join(encoded_dir, 'encoded_dev_%s.npy' % self.name))
        return encoded_train, encoded_dev

    def vis_model(self):
        pass