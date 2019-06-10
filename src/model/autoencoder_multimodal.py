import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import plot_model
from src.model.autoencoder import AutoEncoder


class AutoEncoderMultimodal(AutoEncoder):
    def __init__(self, name, input_dim_A, input_dim_V1, input_dim_V2, input_dim_V3, input_dim_V4, noisy=True, sparse=False):
        AutoEncoder.__init__(self, name, input_dim_A+input_dim_V1+input_dim_V2+input_dim_V3, noisy=noisy, sparse=sparse)
        self.load_basic()
        self.name = '%s_hidden%.2f_batch%d_epoch%d_noise%s' % (name, self.hidden_ratio, self.batch_size, self.epochs, self.noise)
        self.dimension_A = input_dim_A
        self.dimension_V = input_dim_V1 + input_dim_V2 + input_dim_V3 + input_dim_V4
        self.dimension_V1 = input_dim_V1
        self.dimension_V2 = input_dim_V2
        self.dimension_V3 = input_dim_V3
        self.dimension_V4 = input_dim_V4
        self.decoder_A = None
        self.decoder_V1 = None
        self.decoder_V2 = None
        self.decoder_V3 = None
        self.decoder_V4 = None

    def build_model(self):
        if not os.path.isdir(os.path.join(self.save_dir, self.name)):
            os.mkdir(os.path.join(self.save_dir, self.name))
            self.fitted = False
        else:
            self.fitted = True
        
        hidden_dim = int((self.dimension_A + self.dimension_V) * self.hidden_ratio / 4)

        input_data_A = Input(shape=(self.dimension_A, ), name='audio_input')
        input_data_V1 = Input(shape=(self.dimension_V1, ), name='facial_input')
        input_data_V2 = Input(shape=(self.dimension_V2, ), name='gaze_input')
        input_data_V3 = Input(shape=(self.dimension_V3, ), name='pose_input')
        input_data_V4 = Input(shape=(self.dimension_V4, ), name='action_input')
        encoded_input = Input(shape=(hidden_dim, ))
        
        encoded_A = Dense(int(self.dimension_A * self.hidden_ratio), 
                        activation='relu', kernel_initializer='he_uniform',
                        name='audio_encoded')(input_data_A)
        encoded_V1 = Dense(int(self.dimension_V1 * self.hidden_ratio), 
                        activation='relu', kernel_initializer='he_uniform',
                        name='facial_encoded')(input_data_V1)
        encoded_V2 = Dense(int(self.dimension_V2 * self.hidden_ratio), 
                        activation='relu', kernel_initializer='he_uniform',
                        name='gaze_encoded')(input_data_V2)
        encoded_V3 = Dense(int(self.dimension_V3 * self.hidden_ratio),
                        activation='relu', kernel_initializer='he_uniform',
                        name='pose_encoded')(input_data_V3)
        encoded_V4 = Dense(int(self.dimension_V4 * self.hidden_ratio), 
                        activation='relu', kernel_initializer='he_uniform',
                        name='action_encoded')(input_data_V4)

        shared = Concatenate(axis=1, name='concat')([encoded_A, encoded_V1, encoded_V2, encoded_V3, encoded_V4])
        if self.sparse:
            encoded = Dense(hidden_dim, 
                        activation='relu',
                        activity_regularizer=self.sparse_regularizer,
                        name='shared_repres')(shared)
        else:
            encoded = Dense(hidden_dim, 
                        activation='relu',
                        name='shared_repres')(shared)
        
        decoded_A = Dense(int(self.dimension_A * self.hidden_ratio), 
                        activation='relu', name='audio_decoded')(encoded)
        decoded_V1 = Dense(int(self.dimension_V1 * self.hidden_ratio), 
                        activation='relu', name='facial_decoded')(encoded)
        decoded_V2 = Dense(int(self.dimension_V2 * self.hidden_ratio), 
                        activation='relu', name='gaze_decoded')(encoded)
        decoded_V3 = Dense(int(self.dimension_V3 * self.hidden_ratio), 
                        activation='relu', name='pose_decoded')(encoded)
        decoded_V4 = Dense(int(self.dimension_V4 * self.hidden_ratio), 
                        activation='relu', name='action_decoded')(encoded)

        decoded_A = Dense(self.dimension_A, activation='sigmoid',
                        name='audio_recon')(decoded_A)
        decoded_V1 = Dense(self.dimension_V1, activation='sigmoid',
                        name='facial_recon')(decoded_V1)
        decoded_V2 = Dense(self.dimension_V2, activation='sigmoid',
                        name='gaze_recon')(decoded_V2)
        decoded_V3 = Dense(self.dimension_V3, activation='sigmoid',
                        name='pose_recon')(decoded_V3)
        decoded_V4 = Dense(self.dimension_V4, activation='sigmoid',
                        name='action_recon')(decoded_V4)

        self.autoencoder = Model(inputs=[input_data_A, 
                            input_data_V1, input_data_V2, input_data_V3, input_data_V4], 
                            outputs=[decoded_A, 
                            decoded_V1, decoded_V2, decoded_V3, decoded_V4])
        self.encoder = Model(inputs=[input_data_A, 
                            input_data_V1, input_data_V2, input_data_V3, input_data_V4], 
                            outputs=encoded)
        self.decoder_A = Model(inputs=encoded_input, 
                            outputs=self.autoencoder.get_layer('audio_recon')(
                                self.autoencoder.get_layer('audio_decoded')(
                                encoded_input)))
        self.decoder_V1 = Model(inputs=encoded_input, 
                            outputs=self.autoencoder.get_layer('facial_recon')(
                                self.autoencoder.get_layer('facial_decoded')(
                                encoded_input)))
        self.decoder_V2 = Model(inputs=encoded_input, 
                            outputs=self.autoencoder.get_layer('gaze_recon')(
                                self.autoencoder.get_layer('gaze_decoded')(
                                encoded_input)))
        self.decoder_V3 = Model(inputs=encoded_input, 
                            outputs=self.autoencoder.get_layer('pose_recon')(
                                self.autoencoder.get_layer('pose_decoded')(
                                encoded_input)))
        self.decoder_V4 = Model(inputs=encoded_input, 
                            outputs=self.autoencoder.get_layer('action_recon')(
                                self.autoencoder.get_layer('action_decoded')(
                                encoded_input)))

        # configure model
        # two combo ['adam' + 'mse] ['adadelta', 'binary_crossentropy']
        self.autoencoder.compile(optimizer='adam', loss='mse')
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

        plot_model(self.autoencoder, show_shapes=True, to_file=os.path.join(self.save_dir, self.name, 'bimodalV_SDAE.png'))

    def train_model(self, X_train_A, X_train_V, X_dev_A, X_dev_V):
        if self.fitted:
            print("\nmodel already trained ---", self.name)
            self.load_model()
            return 
        
        X_train_V1, X_train_V2, X_train_V3, X_train_V4 = self.separate_V(X_train_V)
        X_dev_V1, X_dev_V2, X_dev_V3, X_dev_V4 = self.separate_V(X_dev_V)

        # normalization to [0,1] for sigmoid
        X_train_A = minmax_scale(X_train_A)
        X_train_V2 = minmax_scale(X_train_V2)
        X_train_V3 = minmax_scale(X_train_V3)
        X_train_V4 = minmax_scale(X_train_V4)
        X_dev_A = minmax_scale(X_dev_A)
        X_dev_V2 = minmax_scale(X_dev_V2)
        X_dev_V3 = minmax_scale(X_dev_V3)
        X_dev_V4 = minmax_scale(X_dev_V4)

        if self.noisy:
            X_train_A_noisy = self.add_noise(X_train_A, self.noise)
            X_train_V1_noisy = self.add_noise(X_train_V1, self.noise)
            X_train_V2_noisy = self.add_noise(X_train_V2, self.noise)
            X_train_V3_noisy = self.add_noise(X_train_V3, self.noise)
            X_train_V4_noisy = self.add_noise(X_train_V4, self.noise)
            X_dev_A_noisy = self.add_noise(X_dev_A, self.noise)
            X_dev_V1_noisy = self.add_noise(X_dev_V1, self.noise)
            X_dev_V2_noisy = self.add_noise(X_dev_V2, self.noise)
            X_dev_V3_noisy = self.add_noise(X_dev_V3, self.noise)
            X_dev_V4_noisy = self.add_noise(X_dev_V4, self.noise)
        else:
            X_train_A_noisy = X_train_A
            X_train_V1_noisy = X_train_V1
            X_train_V2_noisy = X_train_V2
            X_train_V3_noisy = X_train_V3
            X_train_V4_noisy = X_train_V4
            X_dev_A_noisy = X_dev_A
            X_dev_V1_noisy = X_dev_V1
            X_dev_V2_noisy = X_dev_V2
            X_dev_V3_noisy = X_dev_V3
            X_dev_V4_noisy = X_dev_V4

        assert X_train_A_noisy.shape == X_train_A.shape
        assert X_train_V1_noisy.shape == X_train_V1.shape
        assert X_train_V2_noisy.shape == X_train_V2.shape
        assert X_train_V3_noisy.shape == X_train_V3.shape
        assert X_train_V4_noisy.shape == X_train_V4.shape
        assert X_dev_A_noisy.shape == X_dev_A.shape
        assert X_dev_V1_noisy.shape == X_dev_V1.shape
        assert X_dev_V2_noisy.shape == X_dev_V2.shape
        assert X_dev_V3_noisy.shape == X_dev_V3.shape
        assert X_dev_V4_noisy.shape == X_dev_V4.shape

        csv_logger = CSVLogger(os.path.join(self.save_dir, self.name, "logger.csv"))
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, self.name, "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [csv_logger, checkpoint]

        self.autoencoder.fit([X_train_A_noisy, 
                            X_train_V1_noisy, X_train_V2_noisy, 
                            X_train_V3_noisy, X_train_V4_noisy],
                            [X_train_A, X_train_V1, X_train_V2,
                            X_train_V3, X_train_V4],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            validation_data=([X_dev_A_noisy, 
                                        X_dev_V1_noisy, X_dev_V2_noisy, 
                                        X_dev_V3_noisy, X_dev_V4_noisy],
                                        [X_dev_A, X_dev_V1, X_dev_V2, 
                                        X_dev_V3, X_dev_V4]),
                            callbacks=callbacks_list)
        print("\nmodel trained and saved ---", self.name)
        self.save_model()

    def encode(self, X_1_A, X_1_V, X_2_A, X_2_V):
        """encode bimodal input to latent representation
        """
        X_1_V1, X_1_V2, X_1_V3, X_1_V4 = self.separate_V(X_1_V)
        X_2_V1, X_2_V2, X_2_V3, X_2_V4 = self.separate_V(X_1_V)
        encoded_train = self.encoder.predict([X_1_A, X_1_V1, X_1_V2, X_1_V3, X_1_V4])
        encoded_dev = self.encoder.predict([X_2_A, X_2_V1, X_2_V2, X_2_V3, X_2_V4])
        self.save_representation(encoded_train, encoded_dev)
    
    def decode(self, encoded_pre):
        """decode latent representation to bimodal input
        """
        decoded_input_A = self.decoder_A.predict(encoded_pre)
        decoded_input_V1 = self.decoder_V1.predict(encoded_pre)
        decoded_input_V2 = self.decoder_V2.predict(encoded_pre)
        decoded_input_V3 = self.decoder_V3.predict(encoded_pre)
        decoded_input_V4 = self.decoder_V4.predict(encoded_pre)
        return decoded_input_A, decoded_input_V1, decoded_input_V2, decoded_input_V3, decoded_input_V4