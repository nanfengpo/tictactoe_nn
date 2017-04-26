import tensorflow as tf
from collections import namedtuple


TicTacToeGameResult = namedtuple(
    'TicTacToeGameResult',
    'field wins turns stop_signals'
)


def build_game(model1, model2, field_size, win_length, n_games):

    rules = TicTacToeRules(field_size, win_length)

    def do_game_step(field, player, run_flags, results, turn_story, stop_signals):
        turn = TicTacToeTurn(model1, model2, player, field, rules, n_games)
        step = TicTacToeGameStep(
            turn_probs=turn.turn_probs,
            rules=rules,
            field=field,
            player=player,
            results=results,
            run_flags=run_flags,
            turn_story=turn_story,
            n_games=n_games
        )
        updated_stop_signals = stop_signals.write(stop_signals.size(), step.stop_signals)

        return (
            step.updated_field,
            step.next_player,
            step.updated_run_flags,
            step.updated_results,
            step.updated_turn_story,
            updated_stop_signals
        )

    def has_running_games(field, player, run_flags, results, turn_story, stop_signals):
        return TicTacToeGameCondition(run_flags).condition

    initial_field = tf.zeros([n_games, *field_size])
    all_games_run = tf.ones([n_games], dtype=tf.float32)
    empty_turns_story = tf.TensorArray(tf.float32, 0, dynamic_size=True)
    empty_stop_signals = tf.TensorArray(tf.float32, 0, dynamic_size=True)
    game_results = tf.while_loop(
        loop_vars=(initial_field, 1.0,
                   all_games_run, tf.zeros([n_games]),
                   empty_turns_story, empty_stop_signals),
        cond=has_running_games,
        body=do_game_step
    )
    final_field = game_results[0]
    wins = game_results[3]
    final_turn_story = game_results[4].pack()
    final_stop_signals = game_results[5].pack()
    return TicTacToeGameResult(
        final_field, tf.stop_gradient(wins), final_turn_story, final_stop_signals
    )


class TicTacToeTurn:

    def __init__(self, model1, model2, player, field, rules, n_games=-1):
        self.is_first_player = tf.greater(player, 0)
        self.turn_probs = tf.cond(
            self.is_first_player,
            lambda: model1.apply(field),
            lambda: model2.apply(-field),
        )
        self.turn_probs.set_shape([n_games, *rules.field_size])
        self.empty_cells_mask = tf.less(tf.abs(field), 0.5)
        self.turn_probs *= tf.to_float(self.empty_cells_mask)


class TicTacToeGameStep:

    def __init__(self, turn_probs,
                 rules, field, player, run_flags, results, turn_story,
                 n_games=-1):

        self.turns = tf.to_float(tf.equal(
            turn_probs,
            tf.reduce_max(turn_probs, (1, 2), keep_dims=True)
        ))
        self.turns *= tf.reshape(run_flags, [n_games, 1, 1])
        self.updated_turn_story = turn_story.write(turn_story.size(), turn_probs)

        self.updated_field = field + player * self.turns

        self.ttt_result = rules.compute_game_result(field)
        self.updated_run_flags = run_flags * tf.to_float(self.ttt_result.non_finished)
        self.stop_signals = tf.to_float(tf.logical_and(
            run_flags > 0.5,
            self.updated_run_flags < 0.5
        ))
        self.updated_results = results + run_flags * self.ttt_result.who_wins

        self.next_player = tf.convert_to_tensor(-player)


class TicTacToeGameCondition:

    def __init__(self, run_flags):
        self.condition = tf.greater(tf.reduce_sum(run_flags), 0.5)


class TicTacToeRules:

    def __init__(self, field_size, win_length):
        assert win_length <= min(field_size)
        self.field_size = field_size
        self.n_cells = field_size[0] * field_size[1]
        self.win_length = win_length

        diagonal_win_kern = tf.eye(win_length)
        self.win_kernels = {
            'vertical': tf.ones([win_length, 1, 1, 1]),
            'horizontal': tf.ones([1, win_length, 1, 1]),
            'main_diag': tf.reshape(diagonal_win_kern, [win_length, win_length, 1, 1]),
            'collat_diag': tf.reshape(diagonal_win_kern[::-1], [win_length, win_length, 1, 1]),
        }

    def compute_game_result(self, field):
        return TicTacToeResult(field, self)


class TicTacToeResult:

    def __init__(self, field, ttt_rules: TicTacToeRules):
        field = tf.expand_dims(field, 3)

        self.n_filled_cells = tf.reduce_sum(tf.abs(field), axis=(1, 2, 3))
        self.is_not_filled = tf.not_equal(self.n_filled_cells, ttt_rules.n_cells)
        self.does_not_become_filled = tf.not_equal(self.n_filled_cells, ttt_rules.n_cells-1)

        kernels = ttt_rules.win_kernels
        self.vert_strikes = tf.nn.conv2d(field, kernels['vertical'], [1] * 4, 'VALID')
        self.horz_strikes = tf.nn.conv2d(field, kernels['horizontal'], [1] * 4, 'VALID')
        self.diag_strikes = tf.nn.conv2d(field, kernels['main_diag'], [1] * 4, 'VALID')
        self.transp_strikes = tf.nn.conv2d(field, kernels['collat_diag'], [1] * 4, 'VALID')
        self.pos_max_strike_length = tf.reduce_max(tf.pack([
            tf.reduce_max(self.vert_strikes, (1, 2, 3)),
            tf.reduce_max(self.horz_strikes, (1, 2, 3)),
            tf.reduce_max(self.diag_strikes, (1, 2, 3)),
            tf.reduce_max(self.transp_strikes, (1, 2, 3)),
        ], axis=0), 0)
        self.neg_max_strike_length = tf.reduce_max(tf.pack([
            tf.reduce_max(-self.vert_strikes, (1, 2, 3)),
            tf.reduce_max(-self.horz_strikes, (1, 2, 3)),
            tf.reduce_max(-self.diag_strikes, (1, 2, 3)),
            tf.reduce_max(-self.transp_strikes, (1, 2, 3)),
        ], axis=0), 0)
        self.max_strike_length = tf.reduce_max(
            [self.pos_max_strike_length, self.neg_max_strike_length],
            axis=0
        )

        self.has_not_win = tf.less(self.max_strike_length, ttt_rules.win_length - 0.5)

        self.non_finished = tf.logical_and(self.does_not_become_filled, self.has_not_win)

        where_first_wins = self.pos_max_strike_length > self.neg_max_strike_length
        self.who_wins = tf.select(
            where_first_wins,
            tf.ones_like(where_first_wins, dtype=tf.float32),
            -tf.ones_like(where_first_wins, dtype=tf.float32),
        )
        self.who_wins *= tf.to_float(tf.logical_not(self.has_not_win))

