from tictactoe_nn.models import TicTacToe, CombinedModel
from tictactoe_nn.training import TrainingObjective
import tensorflow as tf
import numpy as np
import os
from tictactoe_nn.utils import flatten_values
from ._train_config import Config


config = Config()
tf.get_default_graph().seed = config.seed


target_mdl = TicTacToe(config.variables_scope, config.layers_def)


opponent_models = [
    TicTacToe(opp_def.variables_scope, opp_def.layers_def)
    for opp_def in config.opponents
]
opponent = CombinedModel(*opponent_models)

train_obj = TrainingObjective(target_mdl, opponent,
                              field_size=config.field_size,
                              win_length=config.win_length,
                              n_games=config.n_games)
optimizer = tf.train.AdamOptimizer(config.learning_rate)

train_op = optimizer.minimize(train_obj.total_loss,
                              var_list=target_mdl.get_var_list())


sess = tf.Session()
target_mdl.init_vars(sess)

for opp_mdl, opp_def in zip(opponent_models, config.opponents):
    opp_mdl.load_vars(sess, os.path.join(config.models_root, opp_def.save_path))

sess.run(tf.variables_initializer([
    *flatten_values(optimizer._slots),
    *optimizer._get_beta_accumulators()
]))


class MovingAverage:

    def __init__(self, n_values):
        self.n_values = n_values
        self._values = []
        self._sum = 0

    def add(self, value):
        if not np.isfinite(value):
            return
        self._values.append(value)
        self._sum += value
        if len(self._values) > self.n_values:
            self._sum -= self._values.pop(0)

    def get(self):
        if len(self._values) == 0:
            return 0
        return self._sum / len(self._values)


def save_trained_model():
    mdl_dir = os.path.join(config.models_root, config.save_path, '')
    os.makedirs(mdl_dir)
    target_mdl.save_vars(sess, mdl_dir, config.n_iters)


def run_train(n_iters):
    print('Started')
    moving_loss = MovingAverage(100)
    moving_n_wins = MovingAverage(100)
    moving_n_lost = MovingAverage(100)
    moving_n_draft = MovingAverage(100)
    for i in range(n_iters):
        _, loss, cr_wins, zr_wins = sess.run([
            train_op,
            train_obj.total_loss,
            train_obj.crosses_result.wins,
            train_obj.zeros_result.wins,
            # turns_t,
            # fields_t
        ])
        # mtt = mtt[:, 0, ..., 0]
        n_wins = np.sum(cr_wins == +1) + np.sum(zr_wins == -1)
        n_lost = np.sum(cr_wins == -1) + np.sum(zr_wins == +1)
        n_draft = np.sum(cr_wins == 0) + np.sum(zr_wins == 0)

        moving_loss.add(loss)
        moving_n_wins.add(n_wins)
        moving_n_lost.add(n_lost)
        moving_n_draft.add(n_draft)

        print('{}: {:.5f}/{:.5f}, win={}/{:.2f} lost={}/{:.2f} draft={}/{:.2f}'.format(
            i+1,
            loss, moving_loss.get(),
            n_wins, moving_n_wins.get(),
            n_lost, moving_n_lost.get(),
            n_draft, moving_n_draft.get(),
        ))

        if n_wins+n_lost+n_draft != (cr_wins.size+zr_wins.size):
            print('AAAA!!!')
            print(cr_wins)
            print(zr_wins)
            break
        if moving_n_wins.get() == (cr_wins.size+zr_wins.size):
            print('PWNED')
            break

run_train(config.n_iters)

