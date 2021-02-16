import numpy as np
from scipy.stats import t as stats_t


def make_t_test(n: int):
    sn = np.sqrt(2 / n)

    def test(a: np.array, b: np.array):
        mean_a, var_a = np.moveaxis(a, 0, -1)
        mean_b, var_b = np.moveaxis(b, 0, -1)

        m = mean_a - mean_b
        s = np.sqrt((var_a + var_b) / 2) * sn

        t = m / s

        return 1 - stats_t.sf(np.abs(t).astype(np.float64), n-1) * 2

    return test
