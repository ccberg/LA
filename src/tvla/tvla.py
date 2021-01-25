import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm, ttest_ind
from tqdm import tqdm

from src.data.ctable import ctable_mv
from src.data.dummy.trace_generation import gen_trace, gen_mask_trace
from src.tools.stat_moment import get_mvs
from src.tvla.t import make_t_test


def device_fails(left: np.array, right: np.array):
    return np.array(left * right).any()


def tvla(test, left, right, p=.95, debug=False):
    """
    Applies some given statistical test against the given samples.

    @param test: statistical test function that takes two samples and returns test values.
    @param left: Four sample sets containing s samples representing a power trace of length t.
    @param right: Two sample sets containing s samples representing a power trace of length t.
    @param p: Decide whether a device fails using (100 * p)%-confidence interval.
    @param debug: Set to True if this function should throw Exceptions upon improperly structured input.
    @return: Whether the device fails on (A against A, A against B).
    """
    if debug:
        assert len(left) == 4
        assert len(right) == 2

    # Four different samples from distribution A.
    a, b, c, d = left
    # Two different samples from distribution B.
    x, y = right

    # Test A against A.
    aa1 = test(a, c, p)
    aa2 = test(b, d, p)

    print(aa1, aa2)

    # Test A against B.
    ab1 = test(a, x, p)
    ab2 = test(b, y, p)

    return device_fails(aa1, aa2), device_fails(ab1, ab2)


# Test functions

def stats_t_test(a, b, p=.95):
    return ttest_ind(a, b, equal_var=False)[1] > p


def gen_rvs(mean: float = 5):
    return lambda: np.array([norm.rvs(loc=mean, scale=.1, size=50) for _ in range(50)])


def gen_ctable(sample):
    return np.array([np.bincount(t, minlength=256) for t in sample])


def bench(test, gen_one, gen_two, total=20, progress=False):
    acc = np.array([(False, False)] * total)
    it = range(total)
    if progress:
        it = tqdm(it)

    for ix in it:
        acc[ix] = tvla(test, [gen_one() for _ in range(4)], (gen_two(), gen_two()))

    return np.array(acc).sum(axis=0)


if __name__ == '__main__':
    trc = np.array([round(np.random.uniform(0, 255)) for _ in range(100000)])
    trc_mv = ctable_mv(np.bincount(trc, minlength=256), len(trc))

    # Compare with actual mean and variance
    assert_almost_equal(trc_mv, (trc.mean(), trc.var()))

    # TVLA check for default $t$-test.

    tvla(stats_t_test, [gen_rvs(5)() for _ in range(4)], (gen_rvs(6)(), gen_rvs(6)()))

    # TVLA check for $t$-test with mean and variance as input

    mvs_a = np.moveaxis(get_mvs(np.array([gen_rvs(5)() for _ in range(4)])), 1, -1)
    mvs_b = np.moveaxis(get_mvs(np.array([gen_rvs(6)() for _ in range(2)])), 1, -1)

    tvla(make_t_test(350), mvs_a, mvs_b)

    print("TVLA - number of failing devices: \t[A vs. A, A vs. B]")
    print(f"TVLA for mu = 5 & mu = 5 (rvs): \t{bench(stats_t_test, gen_rvs(5), gen_rvs(5))}")
    print(f"TVLA for mu = 5 & mu = 5.1 (rvs): \t{bench(stats_t_test, gen_rvs(5), gen_rvs(5.1))}")
    print(f"TVLA for mu = 5 & mu = 100 (rvs): \t{bench(stats_t_test, gen_rvs(5), gen_rvs(100))}")
    print(f"TVLA for demo trace & masked trace: \t{bench(stats_t_test, gen_trace, gen_mask_trace)}")
