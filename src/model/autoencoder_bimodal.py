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
    Bimodal Stacked Denoising Autoencoder (SDAE) to encoder audio-visual data
    ---
    Attributes
    -----------
    name: str
        model name
    dimension_A/V: int
        input A/V data dimensionality
    noisy: bool
        whether or not to involve denoising fashion
    sparse: bool
        whether or not to involve sparsity
    decoder_A/V: keras.models.Model
        keras Model mapping latent representation to A/V input
    ---------------------------------------
    Functions
    -----------
    build_model(): public
        build bimodal stacked denoising autoencoder model
    train_model(): public
        train bimodal stacked denoising autoencoder model
    encode(): public
        encode A/V input to latent representation
    decode(): public
        decode latent representation to A/V input
    """
    def __init__(self, name, input_dim_A, input_dim_V, noisy=True, sparse=False):
        # para name: name of bimodal SDAE
        AutoEncoder.__init__(self, name, input_dim_A+input_dim_V, noisy=noisy, sparse=sparse)
        self.load_basic()
        self.dimension_A = input_dim_A
        self.dimension_V = input_dim_V
        self.decoder_A = None
        self.decoder_V = None

    def build_model(self):
        """build bimodal stacked deep autoencoder
        """
        hidden_dim = int((self.dimension_A + self.dimension_V) * self.hidden_ratio / 4)

        input_data_A = Input(shape=(self.dimension_A, ), name='audio_input')
        input_data_V = Input(shape=(self.dimension_V, ), name='video_input')
        encoded_input = Input(shape=(hidden_dim, ))
        
        encoded_A = Dense(int(self.dimension_A * self.hidden_ratio), 
                        activation='relu', name='audio_encoded')(input_data_A)
        encoded_V = Dense(int(self.dimension_V * self.hidden_ratio), 
                        activation='relu', name='video_encoded')(input_data_V)

        shared = Concatenate(axis=1, name='concat')([encoded_A, encoded_V])
        if self.sparse:
            encoded = Dense(hidden_dim, 
                        activation='relu',
                        activity_regularizer=self._sparse_regularizer,
                        name='shared_repres')(shared)
        else:
            encoded = Dense(hidden_dim, 
                        activation='relu',
                        name='shared_repres')(shared)
        
        decoded_A = Dense(int(self.dimension_A * self.hidden_ratio), 
                        activation='relu', name='audio_decoded')(encoded)
        decoded_V = Dense(int(self.dimension_V * self.hidden_ratio), 
                        activation='relu', name='video_decoded')(encoded)

        decoded_A = Dense(self.dimension_A, activation='sigmoid',
                        name='audio_recon')(decoded_A)
        decoded_V = Dense(self.dimension_V, activation='sigmoid',
                        name='video_recon')(decoded_V)

        self.autoencoder = Model(inputs=[input_data_A, input_data_V], outputs=[decoded_A, decoded_V])
        self.encoder = Model(inputs=[input_data_A, input_data_V], outputs=encoded)
        self.decoder_A = Model(inputs=encoded_input, 
                            outputs=self.autoencoder.get_layer('audio_recon')(
                                self.autoencoder.get_layer('audio_decoded')(
                                encoded_input)))
        self.decoder_V = Model(inputs=encoded_input, 
                            outputs=self.autoencoder.get_layer('video_recon')(
                                self.autoencoder.get_layer('video_decoded')(
                                encoded_input)))

        # configure model
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        print("--" * 20)
        print("autoencoder")
        print(self.autoencoder.summary())
        print("--" * 20)
        print("encoder")
        print(self.encoder.summary())
        print("--" * 20)
        print("decoder (A)")
        print(self.decoder_A.summary())
        print("--" * 20)
        print("decoder (V)")
        print(self.decoder_V.summary())

        plot_model(self.autoencoder, show_shapes=True, to_file='./images/models/autoencoder_bimodal.png')

    def train_model(self, X_train_A, X_train_V, X_dev_A, X_dev_V):
        """train bimodal stacked deep autoencoder
        """
        if self.noisy:
            X_train_A_noisy = X_train_A + np.random.normal(loc=0.5, scale=0.5, size=X_train_A.shape)
            X_train_V_noisy = X_train_V + np.random.normal(loc=0.5, scale=0.5, size=X_train_V.shape)
            X_dev_A_noisy = X_dev_A + np.random.normal(loc=0.5, scale=0.5, size=X_dev_A.shape)
            X_dev_V_noisy = X_dev_V + np.random.normal(loc=0.5, scale=0.5, size=X_dev_V.shape)
        else:
            X_train_A_noisy = X_train_A
            X_train_V_noisy = X_train_V
            X_dev_A_noisy = X_dev_A
            X_dev_V_noisy = X_dev_V

        assert X_train_A_noisy.shape == X_train_A.shape
        assert X_train_V_noisy.shape == X_train_V.shape
        assert X_dev_A_noisy.shape == X_dev_A.shape
        assert X_dev_V_noisy.shape == X_dev_V.shape

        self.autoencoder.fit([X_train_A_noisy, X_train_V_noisy],
                            [X_train_A, X_train_V],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            validation_data=(
                                [X_dev_A_noisy, X_dev_V_noisy],
                                [X_dev_A, X_dev_V]))
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
        [decoded_input_A, decoded_input_V] = self.decoder.predict(encoded_pre)
        return decoded_input_A, decoded_input_V