import importlib.resources

import numpy as np
from tensorflow import keras

import gaze_predictor.data
from gaze_predictor.base_network import NeuralNetwork

nn_configuration = {
    'epochs': 20,  # number of epochs
    'batch_size': 32,  # size of the batch
    'verbose': 1,  # set the training phase as verbose
    'optimizer': keras.optimizers.Adam(clipvalue=1.0),  # optimizer
    'metrics': ["root_mean_squared_error"],
    'loss': 'mean_squared_error',  # loss
    'val_split': 0.2,  # validation split: percentage of the training data used for evaluating the loss function
    'input_shape': (15, 20, 1),
    'n_output': 1  # number of outputs = x and y
}


class MultiInputConvNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)
        with importlib.resources.path(gaze_predictor.gaze_predictor.data, input_file) as data_path_X:
            with importlib.resources.path(gaze_predictor.gaze_predictor.data, output_file) as data_path_y:
                self._load_data(data_path_X, data_path_y)

    def _load_data(self, input_path, output_path, flatten=False):
        # TODO how?
        data = np.load(input_path)
        X = data['arr_0']

        data = np.load(output_path)
        y = data['arr_0']

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

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(input_shape=self.config['input_shape'], filters=12, kernel_size=3,
                                strides=1, padding='same', activation='relu', name='Input_Conv'),
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding="valid"),
            keras.layers.Flatten(),
            keras.layers.Dense(40, name='Dense1', activation='relu'),
            keras.layers.Dense(self.config['n_output'], name='Dense2_Out')
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())


class MultiInputConvModule(nn.Module):
    def __init__(self):
        super(MultiInputConvModule, self).__init__()
        self.conv_net = keras.Sequential([
            keras.layers.Conv2D(input_shape=self.config['input_shape'], filters=12, kernel_size=3,
                                strides=1, padding='same', activation='relu', name='Conv1_In'),
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding="valid"),
            keras.layers.Flatten()
        ])
        self.mfd_predictor = keras.Sequential([
            keras.layers.Dense(40 + 1, name='Dense1', activation='relu'),
            keras.layers.Dense(self.config['n_output'], name='Dense2_Out')
        ])

    def forward(self, fm, target_position):
        fm_processed = self.features(fm)
        fc_input = torch.cat((fm_processed, target_position))
        output = self.mfd_predictor(fc_input)
        return output
