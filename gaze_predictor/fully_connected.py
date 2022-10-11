import os

import numpy as np
from tensorflow import keras

from gaze_predictor.base_network import DATA_PATH, NeuralNetwork

nn_configuration = {
    'epochs': 20,  # number of epochs
    'batch_size': 32,  # size of the batch
    'verbose': 1,  # set the training phase as verbose
    'optimizer': keras.optimizers.Adam(clipvalue=1.0),  # optimizer
    'metrics': ["root_mean_squared_error"],
    'loss': 'mean_squared_error',  # loss
    'val_split': 0.2,  # validation split: percentage of the training data used for evaluating the loss function
    'n_input': 300,
    'input_shape': (300,),
    'n_output': 1  # number of outputs = x and y
}


class FCNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)

        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

    def create_model(self):
        print('X shape:', self.X.shape)

        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.Dense(self.config['n_input'], input_shape=self.config['input_shape'], name='Input',
                               kernel_initializer=xavier_initializer),
            keras.layers.Dense(512, name='Hidden1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(32, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(self.config['n_output'], name='Output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())


class FCNetworkHighCapacity(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)

        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

    def create_model(self):
        print('X shape:', self.X.shape)

        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.Dense(self.config['n_input'], input_shape=self.config['input_shape'], name='Input',
                               kernel_initializer=xavier_initializer),
            keras.layers.Dense(1024, name='Hidden1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(256, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(32, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(32, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(16, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(self.config['n_output'], name='Output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())



class FCNetworkPlayerPosInput(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False):
        input_file = 'player_pos.npz'
        output_file = 'mfd.npz'
        # TODO normalize player position?

        super().__init__(name, percent_train, configuration)

        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

    def create_model(self):
        print('X shape:', self.X.shape)

        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.Dense(self.config['n_input'], input_shape=self.config['input_shape'], name='Input',
                               kernel_initializer=xavier_initializer),
            keras.layers.Dense(512, name='Hidden1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(32, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(self.config['n_output'], name='Output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())


class FCNetworkSituationInput(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False, situation_size=7):
        input_file = f'state_{situation_size}x{situation_size}.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)

        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

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
            subject_X = np.concatenate((subject_situation, subject_player_pos), axis=1)

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

    def create_model(self):
        print('X shape:', self.X.shape)

        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.Dense(self.X_train.shape[-1], input_shape=(self.X_train.shape[-1],), name='Input',
                               kernel_initializer=xavier_initializer),
            keras.layers.Dense(512, name='Hidden1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(32, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(self.config['n_output'], name='Output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())


class FCNetworkELU(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)

        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

    def create_model(self):
        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.Dense(self.config['n_input'], input_shape=self.config['input_shape'], name='Input',
                               kernel_initializer=xavier_initializer),
            keras.layers.Dense(512, name='Hidden1', activation='elu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(32, name='Hidden2', activation='elu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(self.config['n_output'], name='Output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())


class FCNetworkDropout(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)
        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

    def create_model(self):
        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.Dense(self.config['n_input'], input_shape=self.config['input_shape'], name='Input',
                               kernel_initializer=xavier_initializer),
            keras.layers.Dense(512, name='Hidden1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dropout(name='DropOut1', rate=0.2),
            keras.layers.Dense(32, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dropout(name='DropOut2', rate=0.2),
            keras.layers.Dense(self.config['n_output'], name='Output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())


class FCNetworkBatchNormalization(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)
        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

    def create_model(self):
        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.Dense(self.config['n_input'], input_shape=self.config['input_shape'], name='Input',
                               kernel_initializer=xavier_initializer),
            keras.layers.Dense(512, name='Hidden1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, name='Hidden2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(self.config['n_output'], name='Output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())
