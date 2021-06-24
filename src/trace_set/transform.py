import numpy as np

from src.tools.balance import balance
from tensorflow.python.keras.utils.np_utils import to_categorical


def reduce_fixed_fixed(x, y):
    """
    Takes 9-class (categorical) hamming weight labels and reduces it to 2 semi-fixed classes.
    """
    numerical = y.argmax(axis=1)
    filter_ixs = numerical != 4

    numerical_reduced = numerical[filter_ixs] > 4
    y2 = to_categorical(numerical_reduced).astype(np.int8)

    return balance(x[filter_ixs], y2)


def reduce_fixed_random(x, y):
    """
    Takes 9-class (categorical) hamming weight labels and reduces it to 2 classes: semi-fixed and random.
    """
    numerical = y.argmax(axis=1)
    filter_ixs = numerical != 4

    numerical_reduced = numerical[filter_ixs] > 4
    y2 = to_categorical(numerical_reduced).astype(np.int8)

    return balance(x[filter_ixs], y2)
