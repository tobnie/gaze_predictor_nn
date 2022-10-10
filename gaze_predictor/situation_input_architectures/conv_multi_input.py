import os
import pickle

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
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
        # data_dir = '../data/'
        subject_dirs = [f for f in os.listdir(data_dir)]

        subject_X1_list = []
        subject_X2_list = []
        subject_y_list = []
        for subject_dir in subject_dirs:
            input_path = data_dir + subject_dir + '/' + input_file
            player_pos_path = data_dir + subject_dir + '/player_pos.npz'
            output_path = data_dir + subject_dir + '/' + output_file

            # print(f'Loading from {input_path}...')
            subject_X1 = np.load(input_path)['arr_0']
            subject_X2 = np.load(player_pos_path)['arr_0']
            subject_X1 = subject_X1.reshape((subject_X2.shape[0], subject_X1.shape[1], subject_X1.shape[2], 1))
            subject_X2 = subject_X2.reshape((subject_X2.shape[0], 2))

            # print(f'Loading from {output_path}...')
            subject_y = np.load(output_path)['arr_0']

            # if only for one subject, immediately return first found subject data
            if subject_specific:
                return subject_X1, subject_X2, subject_y

            subject_X1_list.append(subject_X1)
            subject_X2_list.append(subject_X2)
            subject_y_list.append(subject_y)

        X1 = np.concatenate(subject_X1_list)
        X2 = np.concatenate(subject_X2_list)
        print(f'X1 shape: {X1.shape}')
        print(f'X2 shape: {X2.shape}')
        y = np.concatenate(subject_y_list)

        return X1, X2, y

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
        dataset = dataset.batch(self.config['batch_size'])
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
        print('X1 Input Shape: ', self.X1_train.shape[1:])
        print('X2 Input Shape: ', self.X2_train.shape[1:])

        situation_size = self.X1_train.shape[1]
        state_input = tf.keras.Input((situation_size, situation_size, 1))
        position_input = tf.keras.Input((2,))

        x1 = tf.keras.layers.Conv2D(input_shape=state_input.get_shape(), filters=16, kernel_size=5, strides=1, padding='same',
                                    activation='relu', name='Conv1')(state_input)
        print('First Layer output:', x1)
        x1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(x1)
        x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv2')(x1)
        x1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(x1)
        x1 = tf.keras.layers.Flatten()(x1)

        print('got to end of cnn layer')
        normalization_layer = tf.keras.layers.Normalization(axis=1)
        normalization_layer.adapt(self.X2)
        normalized_pos_input = normalization_layer(position_input)
        x = tf.keras.layers.Concatenate()([x1, normalized_pos_input])
        print('after concatenation:\n', x)
        x = tf.keras.layers.Dense(128, name='Dense1', activation='relu')(x)
        x = tf.keras.layers.Dense(32, name='Dense2', activation='relu')(x)
        output = tf.keras.layers.Dense(1, name='Output')(x)

        print('got to end of dense layer')
        self.model = tf.keras.Model(inputs=[state_input, position_input], outputs=output)
        print(f'Created model for {self.name}:')
        print(self.model.summary())

    def train(self):
        if self.model is None:
            raise RuntimeError('Model must first be created before it can be trained.')

        # compile and fit model
        self.model.compile(optimizer=self.config['optimizer'],
                           loss=self.config['loss'],
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])
        self.history = self.model.fit([self.X1_train, self.X2_train], self.y_train, batch_size=self.config['batch_size'], epochs=self.config['epochs'],
                                      verbose=self.config['verbose'],
                                      validation_split=self.config['val_split'])
        # self.history = self.model.fit(self._input_fn(train=True), batch_size=self.config['batch_size'], epochs=self.config['epochs'],
        #                               verbose=self.config['verbose'],
        #                               validation_split=self.config['val_split'])

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
        self.model = tf.keras.models.load_model(
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


nn_configuration = {
    'epochs': 100,  # number of epochs
    'batch_size': 32,  # size of the batch
    'verbose': 1,  # set the training phase as verbose
    'optimizer': tf.keras.optimizers.Adam(clipvalue=1.0),  # optimizer
    'metrics': ["root_mean_squared_error"],
    'loss': 'mean_squared_error',  # loss
    'val_split': 0.2,  # validation split: percentage of the training data used for evaluating the loss function
    'input_shape': (15, 20, 1),
    'n_output': 1  # number of outputs = x and y
}

multi_input_conv_nn = MultiInputConvNetwork(name='multi_input_conv_nn', configuration=nn_configuration)
multi_input_conv_nn.create_model()
