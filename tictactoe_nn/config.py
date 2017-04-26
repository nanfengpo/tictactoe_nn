from collections import namedtuple
from .utils import UndefinedAttr as _Undefined


OpponentDef = namedtuple('OpponentDef', 'layers_def variables_scope save_path')


class TrainConfig:

    seed = None

    variables_scope = _Undefined()
    save_path = _Undefined()
    layers_def = _Undefined()  # type: list

    field_size = (3, 3)
    win_length = 3

    n_games = 1

    opponents = []

    models_root = '.models'

    learning_rate = 0.01

    n_iters = 10000
