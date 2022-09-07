import importlib.resources

from tensorflow import keras

from gaze_predictor.gaze_predictor.base_network import NeuralNetwork
from gaze_predictor.gaze_predictor.data

nn_configuration = {
    'epochs': 50,  # number of epochs
    'batch_size': 32,  # size of the batch
    'verbose': 1,  # set the training phase as verbose
    'optimizer': keras.optimizers.Adam(clipvalue=1.0),  # optimizer
    'metrics': ["root_mean_squared_error"],
    'loss': 'mean_squared_error',  # loss
    'val_split': 0.2,  # validation split: percentage of the training data used for evaluating the loss function
    'n_input': 300,
    'input_shape': (300,),
    'n_output': 2  # number of outputs = x and y
}


class FCNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None):

        X_path = 'data/input.npz'
        y_path = 'data/output.npz'

        super().__init__(name, percent_train, configuration)
        with importlib.resources.path('gaze_predictor.gaze_predictor.data', y_path) as data_path_X:
            with importlib.resources.path('gaze_predictor.gaze_predictor.data', X_path) as data_path_y:
                self._load_data(data_path_X, data_path_y, flatten=True)

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Dense(N_INPUT=self.config['n_input'], input_shape=self.config['input_shape'], name='Input'),
            keras.layers.Dense(128, name='Hidden1', activation='relu'),
            keras.layers.Dense(16, name='Hidden2', activation='relu'),
            keras.layers.Dense(self.config['n_output'], name='Output')
        ])

        print(f'Created model for {self.__name__}:')
        print(self.model.summary())


