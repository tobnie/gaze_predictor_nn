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
    'input_shape': (15, 20, 1),
    'n_output': 1  # number of outputs = x and y
}


class ConvNetwork(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None, subject_specific=False):
        input_file = 'single_layer_fm.npz'
        output_file = 'mfd.npz'

        super().__init__(name, percent_train, configuration)
        self._load_data(input_file, output_file, flatten=False, subject_specific=subject_specific)

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(input_shape=self.config['input_shape'], filters=16, kernel_size=5,
                                strides=1, padding='same', activation='relu', name='Conv1'),
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv2'),
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, name='Dense1', activation='relu'),
            keras.layers.Dense(16, name='Dense2', activation='relu'),
            keras.layers.Dense(self.config['n_output'], name='Output')
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())
