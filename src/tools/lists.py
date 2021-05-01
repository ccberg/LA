import math
import numpy as np


def moving_average(arr: np.array, window=5):
    """
    Returns moving average.
    Shape of the output is equal to that of the input, padded with np.NaNs.
    """
    res = np.empty(arr.shape)
    res[:] = np.NaN
    cs = np.cumsum(arr, dtype=np.float64)

    padding = window / 2
    start, end = math.floor(padding), arr.shape[0] - math.ceil(padding)

    res[start:end] = cs[window:] - cs[:-window]
    return res / window


def concat(arrays: np.array):
    return np.concatenate(arrays, axis=0)


def random_split(x, fraction):
    """
    Splits the given array in two parts.
    - the first part is approximately of size n * (1 - fraction).
    - the second part is approximately of size n * fraction.
    """
    selection = np.random.random(len(x)) > fraction

    return x[selection], x[np.invert(selection)]


def random_divide(x):
    """
    Divides the given array in two equally sized parts.
    """
    left, right = random_split(x, .5)
    res_size = min(len(left), len(right))

    return left[:res_size], right[:res_size]
