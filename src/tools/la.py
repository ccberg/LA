import numpy as np


def fixed_fixed(x: np.ndarray, hw: np.ndarray):
    """
    Takes 9-class (integer) hamming weight labels and reduces it to 2 semi-fixed integer classes.
    """
    leakage_bit = hw > 4
    drop_mask = hw != 4

    return x[drop_mask], leakage_bit[drop_mask]


def balance(x: np.ndarray, la_bit: np.ndarray):
    num_traces = len(x)

    # Get number of bits where la_bit is 1.
    num_1 = np.sum(la_bit)

    # Get number of bits to remove from majority class
    diff = np.abs(num_traces - 2 * num_1)
    majority_class = num_traces - 2 * num_1 < 0

    # Select some random traces out of the majority class to drop
    bit_ixs = np.where(la_bit == majority_class)[0]
    np.random.shuffle(bit_ixs)
    drop_ixs = bit_ixs[:diff]

    return np.delete(x, drop_ixs, axis=0), np.delete(la_bit, drop_ixs, axis=0)


def shuffle(x, y):
    ix = np.arange(len(x))
    np.random.shuffle(ix)

    return x[ix], y[ix]
