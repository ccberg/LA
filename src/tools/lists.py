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


