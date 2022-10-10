from tensorflow import keras

from gaze_predictor.base_network import NeuralNetwork
from gaze_predictor.recurrent_network import BaseRecurrentNetwork

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


class RecurrentConvNetwork(BaseRecurrentNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False, timesteps=100, stride=20):
        input_file = 'single_layer_fm_seq.npz'
        output_file = 'mfd_seq.npz'

        super().__init__(name, percent_train, configuration, timesteps=timesteps, stride=stride)

        self._load_data(input_file, output_file, flatten=True, subject_specific=subject_specific)

        # flatten data
        print('Train Data before:', self.X_train.shape)
        print('Test Data before:', self.X_test.shape)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.timesteps, 15, 20, 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.timesteps, 15, 20, 1))
        print('X_train after reshape:', self.X_train.shape)
        print('X_test after reshape:', self.X_test.shape)

    def create_model(self):
        xavier_initializer = keras.initializers.GlorotUniform()
        self.model = keras.Sequential([
            keras.layers.ConvLSTM2D(filters=32, kernel_size=5, strides=1, padding='same',
                                    input_shape=self.X_train.shape[1:], name='LSTMConv',
                                    kernel_initializer=xavier_initializer),  # activation is tanh by default
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),
            keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv'),
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, name='Dense1', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(32, name='Dense2', activation='relu', kernel_initializer=xavier_initializer),
            keras.layers.Dense(self.config['n_output'], name='output', kernel_initializer=xavier_initializer)
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())
