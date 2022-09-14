import importlib.resources
from tensorflow import keras
from gaze_predictor.gaze_predictor.base_network import NeuralNetwork
import gaze_predictor.gaze_predictor.data

nn_configuration = {
    'epochs': 20,  # number of epochs
    'batch_size': 32,  # size of the batch
    'verbose': 1,  # set the training phase as verbose
    'optimizer': keras.optimizers.Adam(clipvalue=1.0),  # optimizer
    'metrics': ["root_mean_squared_error"],
    'loss': 'mean_squared_error',  # loss
    'val_split': 0.2,  # validation split: percentage of the training data used for evaluating the loss function
    'input_shape': (15, 20, 1),
    'n_output': 2  # number of outputs = x and y
}


class ConvNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, use_continuous_output=False):
        input_file = 'single_layer_fm.npz'
        if use_continuous_output:
            output_file = 'data/output_continuous.npz'
        else:
            output_file = 'data/output_discrete.npz'

        super().__init__(name, percent_train, configuration)
        with importlib.resources.path(gaze_predictor.gaze_predictor.data, input_file) as data_path_X:
            with importlib.resources.path(gaze_predictor.gaze_predictor.data, output_file) as data_path_y:
                self._load_data(data_path_X, data_path_y)

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(input_shape=self.config['input_shape'], filters=8, kernel_size=3,
                                strides=2, padding='same', activation='relu', name='Input_Conv'),
            keras.layers.Flatten(),
            keras.layers.Dense(16, name='Hidden1', activation='relu'),
            keras.layers.Dense(2, name='Output')
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())
