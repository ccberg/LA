import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats

from src.data.dummy.trace_generation import gen_trace
from src.tools.stat_moment import get_mvs


def make_t_test(n):
    sn = np.sqrt(2 / n)

    def test(a: np.array, b: np.array):
        mean_a, var_a = np.moveaxis(a, 0, -1)
        mean_b, var_b = np.moveaxis(b, 0, -1)

        m = mean_a - mean_b
        s = np.sqrt((var_a + var_b) / 2) * sn

        t = m / s

        return np.abs(t).astype(np.float64)

    return test

