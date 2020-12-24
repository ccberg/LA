import math
from random import choices

import numpy as np
from tqdm import tqdm

from src.tools import cache_np
from src.tools import ASCAD, TraceGroup
from src.tvla.chi2 import chi_squared

MAX_CTABLE_SIZE = 64
DEFAULT_SLIDING_WINDOW_SIZE = 10
ASCAD_NUM_KEYS = 256


def count_sample_point(traces: np.array):
    """
    Simple contingency table creation function that counts occurrences of a power trace value ([0..256] for ASCAD)
    per sample point.

    Used as a reference function, as count(..., window_size = 1) yields the same result.
    """
    return np.array([np.bincount(t, minlength=ASCAD.key_size) for t in np.moveaxis(traces, 0, -1) + (-ASCAD.offset)],
                    dtype=np.uint32)


def reduce_ctable(bin_count: np.array, size_from: int, size_to: int):
    """
    Reduces the size of a given contingency table.
    """
    res = np.zeros(size_to, dtype=np.uint32)
    step = round(size_from / size_to)  # Should be divisible

    for ix in range(size_to):
        res[ix] = np.sum(bin_count[ix * step:(ix + 1) * step])

    return res


def count(traces: np.array, window_size, table_size=MAX_CTABLE_SIZE):
    """
    Creates a contingency table counting the occurrences of a power trace value ([0..256] for ASCAD) per sliding
    window step.

    @param traces: Array of traces.
    @param window_size: The sliding window size. The window step is fixed to 1.
    @param table_size: The size of the resulting contingency table.
    """
    traces_norm = np.moveaxis(traces, 0, -1) + (-ASCAD.offset)
    trace_size = traces_norm.shape[0]

    max_win_ix = trace_size - window_size + 1
    res = np.array([np.zeros(table_size)] * max_win_ix, dtype=np.uint32)

    for ix in range(max_win_ix):
        # Count using the original contingency table.
        orig_table = np.zeros(ASCAD.key_size)
        for t in traces_norm[ix:ix + window_size]:
            orig_table += np.array(np.bincount(t, minlength=ASCAD.key_size), dtype=np.uint32)

        # Reduce the size of the contingency table, so chi^2 will be able to handle it.
        res[ix] = reduce_ctable(orig_table, ASCAD.key_size, table_size)

    return res


def get_max_window(trace_len, win_size):
    """
    Retrieves the number of counted elements as a result of using a sliding window of the provided size on a trace with
    provided size.
    """
    return trace_len - win_size + 1


def balance_count(traces, sw_size, max_len=None):
    """
    Balances the sum of trace contingency tables by randomly incrementing counters using the existing distribution from
    each underrepresented table.

    Useful when dealing with trace sets that are not of the same size.
    """
    num_traces, _ = traces.shape

    split = fill = round(num_traces / 2)
    if max_len is not None:
        # If the trace length to be matched is given.
        fill = math.ceil(max_len / 2)

    def make_ctable(a):
        return count(a, sw_size)

    left, right = make_ctable(traces[split:]), make_ctable(traces[:split])

    num_wins, num_bins = left.shape

    def make_choices(w, k):
        return choices(list(range(num_bins)), weights=w, k=k)

    # Get the number of draws per sliding window index.
    draw_left = (fill - (num_traces - split)) * sw_size
    draw_right = (fill - split) * sw_size

    for sw_ix in range(num_wins):
        # Pull from existing distribution...
        c_left = make_choices(left[sw_ix], draw_left)
        c_right = make_choices(right[sw_ix], draw_right)

        # and add the pulled indexes to the trace counts.
        np.add.at(left[sw_ix], c_left, 1)
        np.add.at(right[sw_ix], c_right, 1)

    return left, right


def get_ctable(tg: TraceGroup, sw_size: int, ks: set) -> (dict, dict):
    max_traces = max([len(tg.profile.filter_traces(i)) for i in ks])

    c_left, c_right = {}, {}
    for k in tqdm(ks, desc="Creating contingency tables   "):
        c_left[k], c_right[k] = balance_count(tg.profile.filter_traces(k), sw_size, max_traces)

    return c_left, c_right


def ctable_cache(tg: TraceGroup, key_byte: int, sw_size: int):
    left, right = cache_np(f"chi_pointwise/ctab_k{key_byte}_w{sw_size}", get_ctable, tg, sw_size, range(256))

    return dict(left), dict(right)


def chi2(counts: tuple, keys: tuple) -> np.array:
    left, right = counts

    assert len(left) > 0
    max_window, _ = left[0].shape

    csp = {}

    progress = tqdm(total=(len(keys[0]) * len(keys[1])), desc=f"Calculating p-values for keys ")
    for k1 in keys[0]:
        csp[k1] = {}
        for k2 in keys[1]:
            csp[k1][k2] = np.zeros(max_window)
            for sp in range(max_window):
                # Error here, at 45465 / 65536
                try:
                    csp[k1][k2][sp] = chi_squared(left[k1][sp], right[k2][sp])
                except KeyError:
                    print(sp)

            progress.update(1)

    progress.close()

    return np.array(list(csp.items()))


def cache_chi2(tg: TraceGroup, key_byte=1, win_size=DEFAULT_SLIDING_WINDOW_SIZE, num_keys=ASCAD_NUM_KEYS):
    cache_name = f"chi_pointwise/chi2_k{key_byte}_w{win_size}_ks{num_keys}"
    key_ranges = (range(num_keys), range(num_keys))
    cont_tables = ctable_cache(tg, key_byte, win_size)

    pvs = cache_np(cache_name, chi2, cont_tables, key_ranges)

    def pad(x):
        s = ASCAD.trace_len - len(x)
        return [*([np.nan] * s), *x]

    # Transform legacy objects.
    if pvs.dtype == np.object:
        pvs = dict(pvs)
        return np.array([[pad(pvs[k1][k2]) for k2 in pvs[k1].keys()] for k1 in pvs.keys()])

    return pvs


if __name__ == '__main__':
    ascad = ASCAD()

    # The sum of bin counts with window = 1 should be equal to the sample-point bin count.
    assert count(ascad.default.profile.filter_traces(0), 1).sum() - \
           count_sample_point(ascad.default.profile.filter_traces(0)).sum() == 0

    tmp = [i.sum() for i in balance_count(ascad.default.profile.filter_traces(3), 10)]

    # Left and right should be balanced.
    assert tmp[1] - tmp[0] == 0
