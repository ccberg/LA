import numpy as np
from numpy import logical_and as and_
from numpy.random import binomial
from scipy.stats import norm
from tqdm import tqdm

from src.trace_set.database import Database
from src.trace_set.set_hw import TraceSetHW
from src.tvla.tvla import Group

STANDARD_NORM = norm(loc=0, scale=1)


def single_rho_test(x, y, progress):
    """
    Single rho-test, as described by the work of Ding et al. (2018):
        "Towards Sound and Optimal Leakage Detection Procedure".
    """
    num_traces, num_sample_points = x.shape

    sp_indexes = range(num_sample_points)
    if progress:
        sp_indexes = tqdm(sp_indexes, "Computing Correlation Coefficients")

    # Correlation coefficient
    rho = np.zeros(num_sample_points, dtype=np.float128)
    for ix in sp_indexes:
        rho[ix] = np.corrcoef(x[:, ix], y)[0, 1]

    # np.log stands for the natural log (ln).
    test_statistic = .5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(num_traces)
    p = 2 * STANDARD_NORM.sf(np.abs(test_statistic.astype(np.float64)))

    return p


def rho_test(x, y, random=False, progress=True):
    """
    rho-test from Ding et al. (2018) applied to the TVLA framework of Goodwill et al. (2011).
    """
    x, y = get_xy(x, y, random)
    m = ab_mask(x)

    x1, y1 = x[~m], y[~m]
    x2, y2 = x[m], y[m]

    p1 = single_rho_test(x1, y1, progress)
    p2 = single_rho_test(x2, y2, progress)

    return np.array([np.max((p1, p2), axis=0)])


def tvla_t_test(x, y, max_order=3, random=False, progress=True):
    """
    t-test from Schneider and Moradi (2016) applied to the TVLA framework of Goodwill et al. (2011).
    """
    x, y = get_xy(x, y, random)
    m = ab_mask(x)

    a1, a2 = x[and_(m, ~y)], x[and_(~m, ~y)]
    b1, b2 = x[and_(m, y)], x[and_(~m, y)]

    ga1, ga2 = Group(a1, max_order, progress), Group(a2, max_order, progress)
    gb1, gb2 = Group(b1, max_order, progress), Group(b2, max_order, progress)

    p = np.ones((max_order + 1, x.shape[1]))

    for order in range(1, max_order + 1):
        _, p1 = ga1.t_test(gb1, order)
        _, p2 = ga2.t_test(gb2, order)

        p[order] = np.max((p1, p2), axis=0)

    return p


def ab_mask(x):
    num_traces = len(x)
    m = np.zeros(num_traces, dtype=bool)
    m[:round(num_traces / 2)] = True
    np.random.shuffle(m)
    return m


def get_xy(x, y, random=False):
    y = y.copy()

    if random:
        np.random.shuffle(y)

    return x, y


if __name__ == '__main__':
    X, Y = TraceSetHW(Database.aisy).profile()

    # t_pvs = tvla_t_test(X, Y)
    rho_pvs = rho_test(X[:1000], Y[:1000])

    # print(np.min(t_pvs), np.argmin(t_pvs))
    print(np.min(rho_pvs), np.argmin(rho_pvs))
