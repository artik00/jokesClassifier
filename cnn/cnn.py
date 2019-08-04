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
        x_test = data[-nb_validation_samples:]
        y_test = labels[-nb_validation_samples:]

        batch_size = 10
        epochs = 1600
        input_vec = Input(shape=(data.shape[1],))
        dense_0 = Dense(20, activation='relu')(input_vec)
        dense_1 = Dense(10, activation='relu')(dense_0)
        dense_2 = Dense(1, activation='sigmoid')(dense_1)
        model = Model(inputs=input_vec, outputs=dense_2)
        model.summary()
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',
                                                     self.f1_m, self.precision_m, self.recall_m])
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

        # 
        # fit the model
        # history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=10, verbose=0)

        # evaluate the model
        loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)

        self.model = model
        print(f"The testing values are: \n Loss: {loss}\n Accuracy: {accuracy}\n F1_score: {f1_score}\n "
              f"Percision: {precision}\n Recall {recall}")

    def recall_m(self, y_true, y_pred):
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + keras.backend.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + keras.backend.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + keras.backend.epsilon()))

