import importlib.resources

from tensorflow import keras
from keras import backend as K

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
    'n_input': 300,
    'input_shape': (300,),
    'n_output': 1  # number of outputs = x and y
}


class RBFLayer(keras.layers.Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


class RBFNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)
        with importlib.resources.path(gaze_predictor.gaze_predictor.data, input_file) as data_path_X:
            with importlib.resources.path(gaze_predictor.gaze_predictor.data, output_file) as data_path_y:
                self._load_data(data_path_X, data_path_y, flatten=True)

    def create_model(self):
        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.Dense(self.config['n_input'], input_shape=self.config['input_shape'], name='Input',
                               kernel_initializer=xavier_initializer),
            RBFLayer(32, 0.5, name='rbf_layer'),
            keras.layers.Dense(self.config['n_output'], name='Output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())
