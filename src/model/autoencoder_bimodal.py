import os
import numpy as np
from keras import metrics
from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import plot_model
from src.model.autoencoder import AutoEncoder


class AutoEncoderBimodal(AutoEncoder):
    """
    Bimodal Deep Denoising Autoencoder (DDAE) to encoder audio-visual data
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
        build bimodal deep denoising autoencoder model
    train_model(): public
        train bimodal deep denoising autoencoder model
    encode(): public
        encode A/V input to latent representation
    decode(): public
        decode latent representation to A/V input
    """
    def __init__(self, name, input_dim_A, input_dim_V, noisy=True, sparse=False):
        # para name: name of bimodal DDAE
        AutoEncoder.__init__(self, name, input_dim_A+input_dim_V, noisy=noisy, sparse=sparse)
        self.load_basic()
        self.name = '%s_hidden%.2f_batch%d_epoch%d_noise%s' % (name, self.hidden_ratio, self.batch_size, self.epochs, self.noise)
        self.dimension_A = input_dim_A
        self.dimension_V = input_dim_V
        self.decoder_A = None
        self.decoder_V = None

    def build_model(self):
        """build bimodal deep deep denoising autoencoder
        """
        if not os.path.isdir(os.path.join(self.save_dir, self.name)):
            os.mkdir(os.path.join(self.save_dir, self.name))
            self.fitted = False
        else:
            self.fitted = True
        
        if self.hidden_ratio != 1.0:
            hidden_dim_A = int(self.dimension_A * self.hidden_ratio)
            hidden_dim_V = int(self.dimension_V * self.hidden_ratio)
            hidden_dim = int((self.dimension_A + self.dimension_V) * self.hidden_ratio / 4)
        else:
            hidden_dim_A = int(self.dimension_A * 0.75)
            hidden_dim_V = int(self.dimension_V * 0.75)
            hidden_dim = int((self.dimension_A + self.dimension_V) * 0.5)

        input_data_A = Input(shape=(self.dimension_A, ), name='audio_input')
        input_data_V = Input(shape=(self.dimension_V, ), name='video_input')
        encoded_input = Input(shape=(hidden_dim, ))
        
        encoded_A = Dense(hidden_dim_A, 
                        activation='relu', kernel_initializer='he_uniform', 
                        name='audio_encoded')(input_data_A)
        encoded_V = Dense(hidden_dim_V, 
                        activation='relu', kernel_initializer='he_uniform', 
                        name='video_encoded')(input_data_V)

        shared = Concatenate(axis=1, name='concat')([encoded_A, encoded_V])
        if self.sparse:
            encoded = Dense(hidden_dim, 
                        activation='relu',
                        activity_regularizer=self.sparse_regularizer,
                        kernel_initializer='he_uniform', 
                        name='shared_repres')(shared)
        else:
            encoded = Dense(hidden_dim, 
                        activation='relu',
                        kernel_initializer='he_uniform', 
                        name='shared_repres')(shared)
        
        decoded_A = Dense(hidden_dim_A, 
                        activation='relu', kernel_initializer='he_uniform', 
                        name='audio_decoded')(encoded)
        decoded_V = Dense(hidden_dim_V, 
                        activation='relu', kernel_initializer='he_uniform', 
                        name='video_decoded')(encoded)

        decoded_A = Dense(self.dimension_A, activation='linear',
                        name='audio_recon')(decoded_A)
        decoded_V = Dense(self.dimension_V, activation='linear',
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
        self.autoencoder.compile(optimizer='adam', 
                                loss='mse',
                                metrics=[metrics.mse, metrics.mse],
                                loss_weights=[0.5, 0.5])
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
        print("--" * 20)

        plot_model(self.autoencoder, show_shapes=True, to_file=os.path.join(self.save_dir, self.name, 'bimodal_DDAE.png'))

    def train_model(self, X_train_A, X_train_V, X_dev_A, X_dev_V):
        """train bimodal deep denoising autoencoder
        """
        if self.fitted:
            print("\nmodel already trained ---", self.name)
            self.load_model()
            return 
        
        X_train_V, _, _, _ = self.separate_V(X_train_V)
        X_dev_V, _, _, _ = self.separate_V(X_dev_V)
        
        X_train_A = np.vstack((X_train_A, X_dev_A))
        X_train_V = np.vstack((X_train_V, X_dev_V))
        
        if self.noisy:
            X_train_A_noisy = self.add_noise(X_train_A, self.noise)
            X_train_V_noisy = self.add_noise(X_train_V, self.noise)
        else:
            X_train_A_noisy = X_train_A
            X_train_V_noisy = X_train_V

        assert X_train_A_noisy.shape == X_train_A.shape
        assert X_train_V_noisy.shape == X_train_V.shape

        csv_logger = CSVLogger(os.path.join(self.save_dir, self.name, "logger.csv"))
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"), monitor='loss', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [csv_logger, checkpoint]

        self.autoencoder.fit([X_train_A_noisy, X_train_V_noisy],
                            [X_train_A, X_train_V],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            callbacks=callbacks_list)
        print("\nmodel trained and saved ---", self.name)
        self.save_model()

    def encode(self, X_train_A, X_train_V, X_dev_A, X_dev_V):
        """encode bimodal input to latent representation
        """
        X_train_V, _, _, _ = self.separate_V(X_train_V)
        X_dev_A, _, _, _ = self.separate_V(X_dev_A)
        
        encoded_train = self.encoder.predict([X_train_A, X_train_V])
        encoded_dev = self.encoder.predict([X_dev_A, X_dev_V])
        self.save_representation(encoded_train, encoded_dev)
        self.decode(encoded_train, encoded_dev)
    
    def decode(self, encoded_train, encoded_dev):
        """decode latent representation to bimodal input
        """
        decoded_train_A = self.decoder_A.predict(encoded_train)
        decoded_dev_A = self.decoder_A.predict(encoded_dev)
        self.save_reconstruction(decoded_train_A, decoded_dev_A, modality=True, no_modality=0)
        decoded_train_V = self.decoder_V.predict(encoded_train)
        decoded_dev_V = self.decoder_V.predict(encoded_dev)
        self.save_reconstruction(decoded_train_V, decoded_dev_V, modality=True, no_modality=1)
