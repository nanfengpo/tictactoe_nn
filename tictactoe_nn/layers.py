import tensorflow as tf
from tictactoe_nn import utils


class Layer:

    def create_variables(self):
        return None

    def apply(self, field, variables):
        raise NotImplementedError

    def apply_transposed(self, field, variables, output_shape):
        raise NotImplementedError


class ConvolutionLayer(Layer):

    def __init__(self, n_inputs, n_outputs, k_size, strides=(1, 1), padding='SAME'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        if isinstance(k_size, int):
            k_size = k_size, k_size
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def create_variables(self):
        name = 'ConvKernel_{}x{}_{}in_{}out'.format(*self.k_size, self.n_inputs, self.n_outputs)
        kernel = tf.get_variable(name, [*self.k_size, self.n_inputs, self.n_outputs])
        return {'kernel': kernel}

    def apply(self, field, variables):
        return tf.nn.conv2d(field, variables['kernel'],
                            strides=[1, *self.strides, 1],
                            padding=self.padding)

    def apply_transposed(self, field, variables, output_shape):
        return tf.nn.conv2d_transpose(
            field, variables['kernel'], output_shape,
            strides=[1, *self.strides, 1]
        )


class ActivationLayer(Layer):

    def __init__(self, function, n_channels):
        if isinstance(function, str):
            if ':' not in function:
                function = 'tensorflow:nn.' + function
            function = utils.load_by_import_path(function)
        self.function = function
        self.n_channels = n_channels

    def create_variables(self):
        name = 'Bias_{}'.format(self.n_channels)
        bias = tf.get_variable(name, [self.n_channels])
        return {'bias': bias}

    def apply(self, field, variables):
        return self.function(field+variables['bias'])

    def apply_transposed(self, field, variables, output_shape):
        assert output_shape[-1] == self.n_channels
        return self.apply(field, variables)


class LayerState:

    def __init__(self, layer):
        self.layer = layer
        self.variables = layer.create_variables()
        self.init_op = tf.variables_initializer(utils.flatten_values(self.variables))

    def init_vars(self, session):
        session.run(self.init_op)

    def apply(self, field):
        return self.layer.apply(field, self.variables)
