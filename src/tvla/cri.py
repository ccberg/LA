import numpy as np
from numpy.random import binomial
from tqdm import tqdm

from src.trace_set.database import Database
from src.trace_set.set_hw import TraceSetHW
from src.trace_set.transform import fixed_fixed
from src.tvla.tvla import Group
from numpy import logical_and as and_


def tvla_cri_run(x, y, max_order, progress):
    num_traces = len(x)
    m = np.zeros(num_traces, dtype=bool)
    m[:round(num_traces / 2)] = True
    np.random.shuffle(m)

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


def get_xy(x, y, random=False):
    y = y.copy()

    if random:
        np.random.shuffle(y)

    return x, y


def tvla_cri(x: np.ndarray, y: np.ndarray, max_order=4, random=False, progress=False):
    x, y = get_xy(x, y, random)
    return tvla_cri_run(x, y, max_order, progress)


def __tvla_cri_p_gradient(x: np.ndarray, y: np.ndarray, order=2, random=False, max_limit: int = None):
    x, y = get_xy(x, y, random)

    num_traces = len(x)
    if max_limit is None:
        max_limit = num_traces

    max_traces = min(num_traces, max_limit)
    limits = np.linspace(2, max_traces, min(200, max_traces - 2)).astype(int)
    pvs = np.ones(len(limits))

    for ix, limit in tqdm(enumerate(limits), total=len(limits), desc="Creating p-gradient"):
        pvs[ix] = np.min(tvla_cri_run(x[:limit], y[:limit], order, False)[order])

    return limits, pvs


def tvla_cri_p_gradient(x: np.ndarray, y: np.ndarray, order=2, random=False, max_limit: int = None, repeat=10):
    limits, pvs = __tvla_cri_p_gradient(x, y, order, random, max_limit)
    res = np.ones((repeat, len(pvs)))
    res[0] = pvs

    for ix in range(1, repeat):
        res[ix] = __tvla_cri_p_gradient(x, y, order, random, max_limit)[1]

    pvs_res = np.nanmean(res, axis=0)
    pvs_res = np.nan_to_num(pvs_res, nan=1)

    return limits, pvs_res


if __name__ == '__main__':
    X, Y = TraceSetHW(Database.aisy).profile()
    print(np.min(tvla_cri(X, Y)))

    print(tvla_cri_p_gradient(X, Y, 1, False, 300, 2))
