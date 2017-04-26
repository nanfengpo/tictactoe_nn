import importlib


def get_nested_attr(obj, attr_path, skip=0):
    if isinstance(attr_path, str):
        attr_path = attr_path.split('.')
    for attr in attr_path[skip:]:
        obj = getattr(obj, attr)
    return obj


def get_import_path(obj):
    return '{}:{}'.format(obj.__module__, obj.__qualname__)


class UndefinedAttr:
    def __get__(self, instance, owner):
        raise NotImplementedError("Attribute not defined")


def load_by_import_path(import_path):
    module_name, attr_path = import_path.split(':')
    obj = importlib.import_module(module_name)
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj


def is_collection(obj):
    if isinstance(obj, str):
        return False
    try:
        iter(obj)
    except (TypeError, ValueError):
        return False
    else:
        return True


def iflatten(*args):
    """
    Arguments may be nested collections - then they will be flattened.
    Also arguments may be not iterable at all - then they simply yielded.
    Examples:
        flatten([1, [2, 3, [4]], 5]) -> [1, 2, 3, 4, 5]
        flatten([1], 2, 3, [[4, 5]]) -> [1, 2, 3, 4, 5]
    :param args: objects, iterable or not
    :return: generator that yields non-iterable objects
    """
    for arg in args:
        if is_collection(arg):
            yield from iflatten(*arg)
        else:
            yield arg


def flatten(*iterables):
    return list(iflatten(*iterables))


def iflatten_values(*args):
    for arg in args:
        if is_collection(arg):
            if isinstance(arg, dict):
                arg = arg.values()
            yield from iflatten_values(*arg)
        else:
            yield arg


def unstack(array, axis):
    if axis < 0:
        axis += array.ndim
    slice_start = (np.s_[:],) * axis
    return [array[slice_start+(i, ...)] for i in range(array.shape[axis])]


def flatten_values(*iterables):
    return list(iflatten_values(*iterables))


def fmap(function, *iterables):
    return list(map(function, *iterables))


def tilestr(*strings, gap=' ', gap_width=1, tostr=str):
    """
    Build string for printing objects in two or more columns.
    Handles objects whose string representation contains many lines.

    Example usage:
    >>> x = np.zeros((3, 4))
    >>> y = np.ones((4, 3))
    >>> print(tilestr(x, y))
    [[ 0.  0.  0.  0.]  [[ 1.  1.  1.]
     [ 0.  0.  0.  0.]   [ 1.  1.  1.]
     [ 0.  0.  0.  0.]]  [ 1.  1.  1.]
                         [ 1.  1.  1.]]
    """
    line_lists = fmap(str.splitlines, map(tostr, strings))

    # normalization: same width within lines list, same length of list
    n_max_lines = max(map(len, line_lists))
    for lines in line_lists:
        list_width = max(map(len, lines))
        for i, line in enumerate(lines):
            lines[i] += ' '*(list_width - len(line))
        lines.extend([' '*list_width]*(n_max_lines - len(lines)))

    gap *= gap_width
    tiled_lines = map(gap.join, zip(*line_lists))
    return '\n'.join(tiled_lines)
