import os
import pickle

import numpy as np
from keras import Model
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.utils import shuffle

from gaze_predictor.base_network import DATA_PATH


class MultiInputConvNetwork:
    SAVE_DIR = '../saved_models/'

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False, situation_size=7):
        input_file = f'state_{situation_size}x{situation_size}.npz'
        output_file = 'mfd.npz'

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

        self._load_data(input_file, output_file, flatten=False, subject_specific=subject_specific)

    def _load_subject_data_and_concat(self, input_file, output_file, subject_specific):
        data_dir = os.getcwd() + DATA_PATH
        subject_dirs = [f for f in os.listdir(data_dir)]

        subject_X_list = []
        subject_y_list = []
        for subject_dir in subject_dirs:
            input_path = data_dir + subject_dir + '/' + input_file
            player_pos_path = data_dir + subject_dir + '/player_pos.npz'
            output_path = data_dir + subject_dir + '/' + output_file

            # print(f'Loading from {input_path}...')
            subject_situation = np.load(input_path)['arr_0']
            subject_player_pos = np.load(player_pos_path)['arr_0']
            subject_player_pos = subject_player_pos.reshape((subject_player_pos.shape[0], 2))
            subject_situation = subject_situation.reshape((subject_situation.shape[0], -1))

            print('Situation data shape:', subject_situation.shape)
            print('player pos data shape:', subject_player_pos.shape)
            subject_X = np.concatenate((subject_situation, subject_player_pos), axis=1)
            print('concatenated data shape:', subject_X.shape)

            # print(f'Loading from {output_path}...')
            subject_y = np.load(output_path)['arr_0']

            # if only for one subject, immediately return first found subject data
            if subject_specific:
                return subject_X, subject_y

            subject_X_list.append(subject_X)
            subject_y_list.append(subject_y)

        X = np.concatenate(subject_X_list)
        y = np.concatenate(subject_y_list)

        return X, y

    def _shuffle_data(self, random_state=0):
        self.X1, self.X2, self.y = shuffle(self.X1, self.X2, self.y, random_state=random_state)

    def _input_fn(self, train=True):
        def generator_train():
            for s1, s2, l in zip(self.X1_train, self.X2_train, self.y_train):
                yield {"input_1": s1, "input_2": s2}, l

        def generator_test():
            for s1, s2, l in zip(self.X1_test, self.X2_test, self.y_test):
                yield {"input_1": s1, "input_2": s2}, l

        generator = generator_train if train else generator_test

        dataset = tf.data.Dataset.from_generator(generator, output_types=({"input_1": tf.int64, "input_2": tf.int64}, tf.float64))
        # dataset = dataset.batch(2)
        return dataset

    def _load_data(self, input_file, output_file, flatten=False, subject_specific=False):
        X1, X2, y = self._load_subject_data_and_concat(input_file, output_file, subject_specific=subject_specific)

        self.X1, self.X2, self.y = X1, X2, y
        self._shuffle_data()

        self.X1_train = self.X1[:int(len(self.X1) * self.percent_train)]
        self.X2_train = self.X2[:int(len(self.X2) * self.percent_train)]
        self.y_train = self.y[:int(len(self.y) * self.percent_train)]

        self.X1_test = self.X1[-int(len(self.X1) * self.percent_test):]
        self.X2_test = self.X2[-int(len(self.X2) * self.percent_test):]
        self.y_test = self.y[-int(len(self.y) * self.percent_test):]

    def create_model(self):
        state_input = keras.layers.Input(shape=self.X1_train.shape[1:])
        region_input = keras.layers.Input(shape=(self.X2_train.shape[-1],))

        x1 = keras.layers.Conv2D(input_shape=self.config['input_shape'], filters=16, kernel_size=5,
                                 strides=1, padding='same', activation='relu', name='Conv1')(state_input),
        x1 = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(x1),
        x1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv2')(x1),
        x1 = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(x1),
        x1 = keras.layers.Flatten()(x1),
        x = keras.layers.concatenate()([x1, region_input])
        x = keras.layers.Dense(64, name='Dense1', activation='relu')(x),
        x = keras.layers.Dense(16, name='Dense2', activation='relu')(x),
        output = keras.layers.Dense(self.config['n_output'], name='Output')(x)

        self.model = Model(inputs=[state_input, region_input], outputs=output)
        print(f'Created model for {self.name}:')
        print(self.model.summary())

    def train(self):
        if self.model is None:
            raise RuntimeError('Model must first be created before it can be trained.')

        # compile and fit model
        self.model.compile(optimizer=self.config['optimizer'],
                           loss=self.config['loss'],
                           metrics=[keras.metrics.RootMeanSquaredError()])
        self.history = self.model.fit(self._input_fn(train=True), batch_size=self.config['batch_size'], epochs=self.config['epochs'],
                                      verbose=self.config['verbose'],
                                      validation_split=self.config['val_split'])

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self._input_fn(train=False))

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