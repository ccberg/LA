import numpy as np

from src.tools.balance import balance
from tensorflow.python.keras.utils.np_utils import to_categorical


# TODO replace with mlp_hw notebook variants

def reduce_fixed_fixed(x, y):
    """
    Takes 9-class (categorical) hamming weight labels and reduces it to 2 semi-fixed classes.
    """
    hamming_weight = np.argmax(y, axis=1)
    filter_ixs = hamming_weight != 4

    is_high = hamming_weight[filter_ixs] > 4
    y2 = to_categorical(is_high).astype(np.int8)

    return balance(x[filter_ixs], y2)


def reduce_fixed_random(x, y):
    """
    Takes 9-class (categorical) hamming weight labels and reduces it to 2 classes: semi-fixed and random.
    """
    hamming_weight = np.argmax(y, axis=1)
    is_random = np.random.binomial(1, .5, len(x)).astype(bool)
    y2 = to_categorical(is_random).astype(np.int8)

    filter_ixs = np.logical_or(hamming_weight < 4, is_random)

    return balance(x[filter_ixs], y2[filter_ixs])


def fixed_fixed(x: np.ndarray, hw: np.ndarray):
    """
    Takes 9-class (integer) hamming weight labels and reduces it to 2 semi-fixed integer classes.
    """
    leakage_bit = hw > 4
    drop_mask = hw != 4

    return x[drop_mask], leakage_bit[drop_mask]
