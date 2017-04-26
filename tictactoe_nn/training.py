import tensorflow as tf
from . import game


class TrainingObjective:

    def __init__(self, trainable_model, opponent_model, field_size, win_length, n_games):
        self.crosses_result = game.build_game(trainable_model, opponent_model,
                                              field_size, win_length, n_games)
        self.qlearn_crosses = QLearning(self.crosses_result, +1)

        self.zeros_result = game.build_game(opponent_model, trainable_model,
                                            field_size, win_length, n_games)
        self.qlearn_zeros = QLearning(self.zeros_result, -1)

        self.total_loss = self.qlearn_crosses.loss + self.qlearn_zeros.loss


class QLearning:

    def __init__(self, game_result: game.TicTacToeGameResult, trained_player, reward_decay=0.9):
        model_turns = game_result.turns[(trained_player+1)%2::2, ...]

        # extract stop signals visible to the model
        n_games = tf.shape(game_result.stop_signals)[1]
        n_turns = tf.shape(game_result.stop_signals)[0]
        stop_signals_reshaped = tf.reshape(game_result.stop_signals, [1, n_turns, n_games, 1])
        model_stop_signals = tf.nn.max_pool(
            stop_signals_reshaped, ksize=[1, 2, 1, 1],
            strides=[1, 2, 1, 1], padding='SAME'
        )[0, :, :, 0]

        max_probs = tf.reduce_max(model_turns, axis=(2, 3), keep_dims=True)
        next_max_probs = tf.concat(1, [max_probs[:, 1:], tf.zeros_like(max_probs[:, :1])])
        rewards = trained_player * game_result.wins * model_stop_signals
        rewards_reshaped = tf.reshape(rewards, [-1, n_games, 1, 1])
        self.loss_map = tf.square(rewards_reshaped + reward_decay * next_max_probs - model_turns)

        self.loss = tf.reduce_mean(self.loss_map)
