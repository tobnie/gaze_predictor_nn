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
    'time_steps': 5,
    'n_input': 300,
    'input_shape': (15, 20, 1),
    'n_output': 1  # number of outputs = mfd
}


class RecurrentNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None):
        input_file = 'single_layer_fm_seq.npz'
        output_file = 'mfd_seq.npz'

        super().__init__(name, percent_train, configuration)
        with importlib.resources.path(gaze_predictor.gaze_predictor.data, input_file) as data_path_X:
            with importlib.resources.path(gaze_predictor.gaze_predictor.data, output_file) as data_path_y:
                self._load_data(data_path_X, data_path_y)

        # flatten data
        print('Train Data before:', self.X_train.shape)
        print('Test Data before:', self.X_test.shape)
        time_steps = self.X.shape[1]
        self.X_train = self.X_train.reshape((self.X_train.shape[0], time_steps, -1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], time_steps, -1))
        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)
        # TODO data format:  5D tensor with shape: (samples, time, rows, cols, channels)

    # TODO run and try it
    def create_model(self):
        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([

            keras.layers.ConvLSTM2D(filters=12, kernel_size=3, strides=1, padding='same', input_shape=(self.X.shape[1], self.config['n_input']), name='input',
                                    kernel_initializer=xavier_initializer),
            keras.layers.Dense(32, name='Hidden1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(self.config['n_output'], name='output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())
