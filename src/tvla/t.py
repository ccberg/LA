import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import t as stats_t
from scipy.stats import ttest_ind


def make_t_test(na: int, nb: int):
    """
    Returns a t-test that takes the sample mean and variance for a list of sample points from A,
    and those for a list of sample points for B.

    Python implementation of the univariate t-test from the work of Schneider & Moradi (2016):
        "Leakage assessment methodology: Extended version"
    """
    def welch_t_test(ma: np.array, va: np.array, mb: np.array, vb: np.array):
        m = ma - mb

        sa = va / na
        sb = vb / nb

        sab = sa + sb
        t = m / np.sqrt(sab)

        # Approximation from paper.
        dof = na + nb

        # Student's t CDF.
        p = 2 * stats_t(df=dof).cdf(-np.abs(t))

        return t, p

    return welch_t_test


if __name__ == '__main__':
    # Test case, comparing the Schneider variant of the t-test to the t-test implementation of scipy.

    def gen_example(mean, trace_num, trace_len=1400):
        return np.random.normal(mean, 2.2, size=(trace_num, trace_len)).astype(int)

    def get_mv(x: np.array):
        return np.array((x.mean(axis=0), x.var(axis=0)))

    num_traces = 10000

    ex_a = gen_example(2, num_traces)
    ex_b = gen_example(2, num_traces)

    ex_a_s = get_mv(ex_a).shape

    res_sp = ttest_ind(ex_a, ex_b, axis=0, equal_var=False)[0]

    test = make_t_test(num_traces, num_traces)
    res_custom = test(*get_mv(ex_a), *get_mv(ex_b))[0]

    assert_almost_equal(res_custom, res_sp, decimal=3)

