import tensorflow as tf
from .layers import LayerState
from .utils import flatten_values


class TicTacToe:

    def __init__(self, scope, layers_def):
        with tf.variable_scope(scope):
            self.layers = []
            for i, lr in enumerate(layers_def, 1):
                with tf.variable_scope('Lr{}'.format(i)):
                    self.layers.append(LayerState(lr))
        self._saver = tf.train.Saver(self.get_var_list())

    def init_vars(self, session):
        for ls in self.layers:
            ls.init_vars(session)

    def save_vars(self, session, path, step=None):
        self._saver.save(session, path, step)

    def load_vars(self, session, path):
        self._saver.restore(session, tf.train.latest_checkpoint(path))

    def apply(self, field):
        field = tf.expand_dims(field, 3)
        for ls in self.layers:
            field = ls.apply(field)
        field = tf.squeeze(field, 3)
        exp_field = tf.exp(field)
        exp_sum = tf.reduce_sum(exp_field, axis=(1, 2), keep_dims=True)
        return exp_field / exp_sum

    def get_var_list(self):
        return flatten_values([ls.variables for ls in self.layers])


class RandomModel:

    def apply(self, field):
        return tf.exp(tf.random_normal(tf.shape(field)))


class CombinedModel:

    def __init__(self, *models):
        self.models = models
        self.random_model = RandomModel()

    def apply(self, field):
        answers = [
            mdl.apply(field[i:i+1])
            for i, mdl in enumerate(self.models)
        ]
        answers.append(self.random_model.apply(field[len(self.models):]))
        return tf.concat(0, answers)
