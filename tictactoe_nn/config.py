import pathlib
from collections import namedtuple
from .utils import UndefinedAttr as _Undefined


OpponentDef = namedtuple('OpponentDef', 'layers_def variables_scope save_path')


class TrainConfig:
    """Default configuration for training"""

    """A random seed to use for variables initialization"""
    seed = None

    """Scope for model's variables"""
    variables_scope = _Undefined()  # type: str

    """Where to save the trained model"""
    save_path = _Undefined()  # type: str

    """A list of tictactoe_nn.layers.Layer objects.
    Defines the structure of the network."""
    layers_def = _Undefined()  # type: list

    """The size of game field. Should be tuple of two integers"""
    field_size = (3, 3)
    """Minimum strike length needed to win the game"""
    win_length = 3

    """Number of games per training batch."""
    n_games = 1

    """
    List of OpponentDef objects.
    Use it if you want to train your model to play against other trained models
    """
    opponents = []

    """
    The root of all saved models. Defaults to <repo-root>/.models
    """
    models_root = str(pathlib.Path(__file__).parents[1].joinpath('.models'))

    learning_rate = 0.01

    """Maximum training iterations."""
    n_iters = 10000
