import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tensorflow import keras

DATA_PATH = '/gaze_predictor/data/'


class NeuralNetwork:
    SAVE_DIR = '../saved_models/'

    def __init__(self, name, percent_train, configuration):
        self.name = name

        # set training and test set sizes
        self.percent_train = percent_train
        self.percent_test = 1 - percent_train
        self.save_path_model = self.SAVE_DIR + name
        self.save_path_history = self.save_path_model + '_history.history'

        self.config = configuration

        # load data
        self.history = None
        self.model = None

    def _load_subject_data_and_concat(self, input_file, output_file, subject_specific):
        data_dir = os.getcwd() + DATA_PATH
        subject_dirs = [f for f in os.listdir(data_dir)]

        subject_X_list = []
        subject_y_list = []
        for subject_dir in subject_dirs:
            input_path = data_dir + subject_dir + '/' + input_file
            output_path = data_dir + subject_dir + '/' + output_file

            # print(f'Loading from {input_path}...')
            subject_X = np.load(input_path)['arr_0']

            # print(f'Loading from {output_path}...')
            subject_y = np.load(output_path)['arr_0']

            # if only for one subject, data for best subject JO03SA
            if subject_specific and subject_dir == 'JO03SA':
                return subject_X, subject_y

            subject_X_list.append(subject_X)
            subject_y_list.append(subject_y)

        X = np.concatenate(subject_X_list)
        y = np.concatenate(subject_y_list)

        return X, y

    def _load_data(self, input_file, output_file, flatten=False, subject_specific=False):

        X, y = self._load_subject_data_and_concat(input_file, output_file, subject_specific=subject_specific)

        if flatten:
            n = X.shape[0]
            X = X.ravel()
            X = X.reshape((n, -1))

        self.X, self.y = X, y
        self._shuffle_data()

        self.X_train = self.X[:int(len(self.X) * self.percent_train)]
        self.y_train = self.y[:int(len(self.y) * self.percent_train)]

        self.X_test = self.X[-int(len(self.X) * self.percent_test):]
        self.y_test = self.y[-int(len(self.y) * self.percent_test):]

    def _shuffle_data(self, random_state=0):
        self.X, self.y = shuffle(self.X, self.y, random_state=random_state)

    def train(self):
        if self.model is None:
            raise RuntimeError('Model must first be created before it can be trained.')

        # compile and fit model
        self.model.compile(optimizer=self.config['optimizer'],
                           loss=self.config['loss'],
                           metrics=[keras.metrics.RootMeanSquaredError()])
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.config['batch_size'], epochs=self.config['epochs'],
                                      verbose=self.config['verbose'],
                                      validation_split=self.config['val_split'])

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)

        print('\nTest {}: {}'.format(self.config['metrics'][0], test_acc))
        print('\nTest loss: {}'.format(test_loss))

        self._plot_rmse()
        self._plot_loss()

        return test_acc, test_loss

    def predict(self, X):
        if self.model is None:
            raise RuntimeError('Model must be trained or loaded before it can make any predictions.')

        return self.model.predict(X)

    def load_model(self, path):
        self.model = keras.models.load_model(
            path, custom_objects=None, compile=True, options=None
        )
        self.history = pickle.load(self.save_path_history)

    def save_model(self):
        self.model.save(self.name + '.model')  # self.save_path_model)
        with open(self.name + '.history', "wb") as f:
            pickle.dump(self.history.history, f)

    def _plot_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.savefig(self.SAVE_DIR + 'plots/' + self.name + '_loss.png')
        plt.savefig(self.name + '_loss.png')
        plt.show()

    def _plot_rmse(self):
        metric_name = self.config['metrics'][0]
        plt.plot(self.history.history[metric_name])
        plt.plot(self.history.history['val_' + metric_name])
        plt.title('model history')
        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.savefig(self.SAVE_DIR + 'plots/' + self.name + '_rmse.png')
        plt.savefig(self.name + '_rmse.png')
        plt.show()
