import numpy as np


def make_t_test(n: int):
    sn = np.sqrt(2 / n)

    def test(a: np.array, b: np.array):
        mean_a, var_a = np.moveaxis(a, 0, -1)
        mean_b, var_b = np.moveaxis(b, 0, -1)

        m = mean_a - mean_b
        s = np.sqrt((var_a + var_b) / 2) * sn

        t = m / s

        return np.abs(t).astype(np.float64)

    return test
