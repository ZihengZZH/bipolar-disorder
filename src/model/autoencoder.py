import os
import json
import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import plot_model


class AutoEncoder():
    """
    Stacked Denoising Autoencoder (SDAE) to encode visual data
    ---
    Attributes
    -----------
    name: str
        model name
    input_dim: int
        input data dimensionality
    noisy: bool
        whether or not to involve denoising fashion
    sparse: bool
        whether or not to involve sparsity
    autoencoder: keras.models.Model
        keras Model mapping input to reconstructed input
    encoder: keras.models.Model
        keras Model mapping input to latent representation
    decoder: keras.models.Model
        keras Model mapping latent representation to input
    dimension: list
        list of dimensionalities in hidden layers
    save_dir: str
        saving directory
    model_config: json.load()
        configuration
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
    def __init__(self, name, input_dim, noisy=True, sparse=False, visual=True):
        # para name: name of SDAE
        self.noisy = noisy
        self.sparse = sparse
        self.visual = visual
        # AE model
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

        self.dimension = [input_dim] * 5
        self.model_config = json.load(open('./config/model.json', 'r'))['autoencoder']
        self.fitted = False
        self.load_basic()
        self.name = '%s_hidden%.2f_batch%d_epoch%d_noise%s' % (name, self.hidden_ratio, self.batch_size, self.epochs, self.noise)
        np.random.seed(1337)
        tf.reset_default_graph()

    def load_basic(self):
        """load basic data and configuration for model
        """
        self.hidden_ratio = self.model_config['hidden_ratio']
        self.learning_rate = self.model_config['learning_rate']
        self.batch_size = self.model_config['batch_size']
        self.epochs = self.model_config['epochs']
        self.noise = self.model_config['noise']
        self.p = self.model_config['p']
        self.beta = self.model_config['beta']
        self.save_dir = self.model_config['save_dir']
        self.dimension[1] = int(self.dimension[0] * self.hidden_ratio)
        self.dimension[2] = int(self.dimension[1] * self.hidden_ratio)
        self.dimension[3] = self.dimension[1]
        self.dimension[4] = self.dimension[0]
        print("\nSDAE initialized and configuration loaded")

    def sparse_regularizer(self, activation_matrix):
        """define the custom regularizer function
        """
        p = 0.01
        beta = 3
        p_hat = K.mean(activation_matrix)
        KLD = p*(K.log(p/p_hat)) + (1-p)*(K.log(1-p/1-p_hat))
        return beta*K.sum(KLD)

    def add_noise(self, X, noise, gaussian=False):
        """add noise (zeros or gaussian)
        """
        if gaussian:
            X_noisy = X + np.random.normal(loc=0.0, scale=0.5, size=X.shape)
        else:
            assert noise <= 0.4, "noise should be not be greater than 0.4"
            idx = np.random.choice(X.shape[1], size=int(X.shape[1] * noise))
            X_noisy = X
            if self.visual:
                X_noisy.iloc[:, idx] = 0.0
            else:
                X_noisy[:, idx] = 0.0
        return X_noisy

    def separate_V(self, X):
        X1 = X.iloc[:, :136] / 1000 # facial 
        X2 = X.iloc[:, 136:142]     # gaze
        X3 = X.iloc[:, 142:148]     # pose
        X4 = X.iloc[:, 148:]        # action
        return X1, X2, X3, X4

    def build_model(self):
        """build stacked denoising autoencoder model
        """
        if not os.path.isdir(os.path.join(self.save_dir, self.name)):
            os.mkdir(os.path.join(self.save_dir, self.name))
            self.fitted = False
        else:
            self.fitted = True
        
        # input placeholder
        input_data = Input(shape=(self.dimension[0], ))
        # encoded input placeholder
        encoded_input = Input(shape=(self.dimension[2], ))

        # encoder part
        encoded = Dense(self.dimension[1], activation='relu', kernel_initializer='he_uniform')(input_data)
        encoded = Dense(self.dimension[2], activation='relu', kernel_initializer='he_uniform', activity_regularizer=self.sparse_regularizer)(encoded) if self.sparse else Dense(self.dimension[2], activation='relu', kernel_initializer='he_uniform')(encoded)
        # decoder part
        decoded = Dense(self.dimension[3], activation='relu', kernel_initializer='he_uniform')(encoded)
        decoded = Dense(self.dimension[4], activation='sigmoid')(decoded)

        # maps input to reconstruction
        self.autoencoder = Model(input_data, decoded)
        # maps input to representation
        self.encoder = Model(input_data, encoded)

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

        plot_model(self.autoencoder, show_shapes=True, to_file=os.path.join(self.save_dir, self.name, 'SDAE.png'))

    def train_model(self, X_train, X_dev):
        """train stacked denoising autoencoder model
        """
        if self.fitted:
            print("\nmodel already trained ---", self.name)
            self.load_model()
            return 
        
        if self.visual:
            X_train, _, _, _ = self.separate_V(X_train)
            X_dev, _, _, _ = self.separate_V(X_dev)
        else:
            X_train = minmax_scale(X_train)
            X_dev = minmax_scale(X_dev)
        
        if self.noisy:
            X_train_noisy = self.add_noise(X_train, self.noise)
            X_dev_noisy = self.add_noise(X_dev, self.noise)
        else:
            X_train_noisy = X_train
            X_dev_noisy = X_dev

        assert X_train_noisy.shape == X_train.shape
        assert X_dev_noisy.shape == X_dev.shape

        csv_logger = CSVLogger(os.path.join(self.save_dir, self.name, "logger.csv"))
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [csv_logger, checkpoint]

        self.autoencoder.fit(X_train_noisy, X_train, 
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            validation_data=(X_dev_noisy, X_dev),
                            callbacks=callbacks_list)
        print("\nmodel trained and saved ---", self.name)
        self.save_model()
    
    def encode(self, X_1, X_2):
        """encode raw input to latent representation
        """
        if self.visual:
            X_1, _, _, _ = self.separate_V(X_1)
            X_2, _, _, _ = self.separate_V(X_2)
        else:
            X_1 = minmax_scale(X_1)
            X_2 = minmax_scale(X_2)
        
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
        self.autoencoder.save_weights(os.path.join(self.save_dir, self.name, 'SDAE.h5'))
        print("\nsaving completed ---", self.name)

    def load_model(self):
        """load stacked denoising autoencoder model from external file
        """
        self.autoencoder.load_weights(os.path.join(self.save_dir, self.name, 'SDAE.h5'))
        print("\nloading completed ---", self.name)

    def save_representation(self, encoded_train, encoded_dev):
        """save encoded representation to external file
        """
        encoded_dir = os.path.join(self.save_dir, self.name)
        np.save(os.path.join(encoded_dir, 'encoded_train'), encoded_train)
        np.save(os.path.join(encoded_dir, 'encoded_dev'), encoded_dev)
    
    def load_presentation(self):
        """load encoded representation from external file
        """
        encoded_dir = os.path.join(self.save_dir, self.name)
        encoded_train = np.load(os.path.join(encoded_dir, 'encoded_train.npy'))
        encoded_dev = np.load(os.path.join(encoded_dir, 'encoded_dev.npy'))
        return encoded_train, encoded_dev

    def save_reconstruction(self, decoded_input_A, decoded_input_V):
        """save reconstructed input
        """
        decoded_dir = os.path.join(self.save_dir, self.name)
        np.save(os.path.join(decoded_dir, 'decoded_A'), decoded_input_A)
        np.save(os.path.join(decoded_dir, 'decoded_V'), decoded_input_V)