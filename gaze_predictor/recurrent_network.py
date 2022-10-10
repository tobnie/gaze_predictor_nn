import os

import numpy as np
from tensorflow import keras

from gaze_predictor.base_network import NeuralNetwork

nn_configuration = {
    'epochs': 20,  # number of epochs
    'batch_size': 32,  # size of the batch
    'verbose': 1,  # set the training phase as verbose
    'optimizer': keras.optimizers.Adam(clipvalue=1.0),  # optimizer
    'metrics': ["root_mean_squared_error"],
    'loss': 'mean_squared_error',  # loss
    'val_split': 0.2,  # validation split: percentage of the training data used for evaluating the loss function
    'time_steps': 5,
    'n_input': 300,
    'input_shape': (300,),
    'n_output': 1  # number of outputs = mfd
}

DATA_PATH = '/gaze_predictor/data/'


class RecurrentNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False, timesteps=100, stride=20):
        input_file = 'single_layer_fm_seq.npz'
        output_file = 'mfd_seq.npz'

        super().__init__(name, percent_train, configuration)

        # lstm specific parameters
        self.timesteps = timesteps
        self.stride = stride
        self.name += f'_timesteps={timesteps}_stride={stride}'

        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

        # flatten data
        print('Train Data before:', self.X_train.shape)
        print('Test Data before:', self.X_test.shape)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.timesteps, -1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.timesteps, -1))
        print('X_train after reshape:', self.X_train.shape)
        print('X_test after reshape:', self.X_test.shape)

    def create_model(self):
        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=(self.X.shape[1], self.config['n_input']), name='input',
                              kernel_initializer=xavier_initializer),
            keras.layers.Dense(128, name='Dense1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(16, name='Dense2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(self.config['n_output'], name='output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())

    def _load_subject_data_and_concat(self, input_file, output_file, subject_specific):
        data_dir = os.getcwd() + DATA_PATH
        subject_dirs = [f for f in os.listdir(data_dir)]

        subject_X_list = []
        subject_y_list = []
        for subject_dir in subject_dirs:
            input_path = data_dir + subject_dir + f'/lstm/timesteps={self.timesteps}_stride={self.stride}/' + input_file
            output_path = data_dir + subject_dir + f'/lstm/timesteps={self.timesteps}_stride={self.stride}/' + output_file

            # print(f'Loading from {input_path}...')
            subject_X = np.load(input_path)['arr_0']

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
