import importlib.resources
from tensorflow import keras
from gaze_predictor.base_network import NeuralNetwork
import gaze_predictor.data

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


class ConvNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)
        with importlib.resources.path(gaze_predictor.gaze_predictor.data, input_file) as data_path_X:
            with importlib.resources.path(gaze_predictor.gaze_predictor.data, output_file) as data_path_y:
                self._load_data(data_path_X, data_path_y)

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(input_shape=self.config['input_shape'], filters=16, kernel_size=5,
                                strides=1, padding='same', activation='relu', name='Input_Conv'),
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),
            keras.layers.Conv2D(filters=32, kernel_size=3,
                                strides=1, padding='same', activation='relu', name='Input_Conv'),
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(40, name='Dense1', activation='relu'),  # TODO number of nodes
            keras.layers.Dense(self.config['n_output'], name='Dense2_Out')
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())