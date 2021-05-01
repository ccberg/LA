import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import t as stats_t
from scipy.stats import ttest_ind


def make_t_test(n: int):
    """
    Returns a t-test that takes the sample mean and variance for a list of sample points from A, and a list of sample
    points for B.

    n should be larger than 1.
    """
    def welch_t_test(ma: np.array, va: np.array, mb: np.array, vb: np.array):
        m = ma - mb

        sa = va / n
        sb = vb / n

        sab = sa + sb
        t = m / np.sqrt(sab)

        dof = n

        p = 2 * stats_t(df=dof).cdf(-np.abs(t))

        return t, p

    return welch_t_test


if __name__ == '__main__':
    def gen_example(mean, trace_num, trace_len=1400):
        return np.random.normal(mean, 2.2, size=(trace_num, trace_len)).astype(int)


    def get_mv(x: np.array):
        return np.array((x.mean(axis=0), x.var(axis=0)))


    num_traces = 100

    ex_a = gen_example(2, num_traces)
    ex_b = gen_example(2, num_traces)

    ex_a_s = get_mv(ex_a).shape

    res_sp = ttest_ind(ex_a, ex_b, axis=0, equal_var=False)[0]

    test = make_t_test(num_traces)
    res_custom = test(get_mv(ex_a), get_mv(ex_b))[0]

    assert_almost_equal(res_custom, res_sp)
