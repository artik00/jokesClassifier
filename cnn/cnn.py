import keras
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pickle


VALIDATION_SPLIT = 0.20


class CNN:

    def __init__(self, data: np.ndarray, labels, debug=None):
        """
        build the CNN
        :param data: the vectors the represent the features the represent in the article
        :param labels: the labels that be used in the cnn model
        """
        self.model = None
        self.loss = None
        self.accuracy = None
        self.precision = None
        self.f1_score = None
        self.recall = None
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
        epochs = 1
        input_vec = Input(shape=(data.shape[1],))
        dense_0 = Dense(20, activation='relu')(input_vec)
        dense_1 = Dense(10, activation='relu')(dense_0)
        dense_2 = Dense(1, activation='sigmoid')(dense_1)
        model = Model(inputs=input_vec, outputs=dense_2)
        model.summary()

        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',
        # self.f1_m, self.precision_m, self.recall_m])
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        acc, metrics = model.evaluate(x_test, y_test, verbose=0)
        if debug:
            self.loss, self.accuracy, self.f1_score, self.precision, \
                self.recall = model.evaluate(x_test, y_test, verbose=0)
            print(f"ACC - {acc}\n Metrics - {metrics} \n  History - {history} ")
        self.model = model
        pickle.dump(model, open("model/model.sav", 'wb'))

    def recall_m(self, y_true, y_pred):
        """
        function to pass to the model for recall calculation
        :param y_true: positive test
        :param y_pred: negative test
        :return: recall value
        """
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + keras.backend.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        """
        function to pass to the model for precision calculation
        :param y_true: positive test
        :param y_pred: negative test
        :return: precision value
        """
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + keras.backend.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        """
        function to pass to the model for F1 calculation
        :param y_true: positive test
        :param y_pred: negative test
        :return: F1§§ value
        """
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + keras.backend.epsilon()))

