import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import MaxPool1D, InputLayer
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
from typing import  List

VALIDATION_SPLIT = 0.20


class CNN:

    def __init__(self, data: np.ndarray, labels):
        self.model = None
        data_to_3d = data.reshape(-1, data.shape[0], data.shape[1] )
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]
        seq_len = data.shape[0]
        #inputs = InputLayer(input_shape=(data.shape[1], 1), dtype='float32')
        # embedding = embedding_layer(inputs)

        # print(embedding.shape)
        # reshape = Reshape((seq_len,EMBEDDING_DIM,1))(embedding)
        # print(reshape.shape)
        num_filters = 512
        filter_sizes = [3, 4, 5]

        # dropout probability
        drop = 0.5
        batch_size = 100
        epochs = 20
        input_vec = Input(shape=(data.shape[1],))
        dense_0 = Dense(20, activation='relu')(input_vec)
        dense_1 = Dense(10, activation='relu')(dense_0)
        dense_2 = Dense(1, activation='sigmoid')(dense_1)
        model = Model(inputs=input_vec, outputs=dense_2)
        model.summary()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # model.summary()
        # embedding_dim = seq_len
        # conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')
        # conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')
        # conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')
        #
        # maxpool_0 = MaxPool1D(pool_size=filter_sizes[0] + 1, strides=None, padding='valid')(conv_0)
        # maxpool_1 = MaxPool1D(pool_size=filter_sizes[1] + 1, strides=None, padding='valid')(conv_1)
        # maxpool_2 = MaxPool1D(pool_size=filter_sizes[2] + 1, strides=None, padding='valid')(conv_2)
        #
        # concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        # flatten = Flatten()(concatenated_tensor)
        # dropout = Dropout(drop)(flatten)
        # output = Dense(units=20, activation='softmax')(dropout)
        #
        # # this creates a model that includes
        # model = Model(inputs=inputs, outputs=output)

        # checkpoint = ModelCheckpoint('weights_cnn_sentece.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        #adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        #model.summary()

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
        #, callbacks=[checkpoint],


        self.model = model
