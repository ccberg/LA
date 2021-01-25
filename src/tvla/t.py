import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats

from src.data.dummy.trace_generation import gen_trace
from src.tools.stat_moment import get_mvs


def make_t_test(n):
    def test(a, b, p):
        ma, sa = a
        mb, sb = b

        s = np.sqrt((sa + sb) / 2)
        t = (ma - mb) / (s * np.sqrt(2 / n))
        dof = 2 * n - 2

        t_abs = np.abs(t).astype(np.float64)
        return stats.distributions.t.sf(t_abs, dof) * 2 > p

    return test


if __name__ == '__main__':
    sps = np.array([gen_trace(5), gen_trace(5)])

    t_mv = [make_t_test(350)(*mv, .95) for mv in np.moveaxis(get_mvs(sps), 0, 1)]
    t_base = np.array(stats.stats.ttest_ind(*sps, equal_var=False)[1] > .95)

    # If the sum of mv = 0, the equality assertion between t_mv and t_base would always pass.
    assert sum(t_mv) > 0

    assert_almost_equal(t_mv, t_base)
