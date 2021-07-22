import numpy as np
from numpy.testing import assert_equal, assert_raises

from src.tools.lists import find
from src.trace_set.database import Database
from src.trace_set.set_hw import TraceSetHW


def window_cxt(trace_cxt, sample_trace, window_margin: int = None):
    start_ix = find(trace_cxt[0], sample_trace)

    win_size = len(sample_trace)
    window = np.array((start_ix, start_ix + win_size))

    if window_margin is None:
        window_margin = win_size

    # Window with context
    cxt = window + np.array((-window_margin, window_margin))

    # Return the window relative to the window context.
    rel_window = window - cxt[0]

    return rel_window, cxt


if __name__ == '__main__':
    ts = TraceSetHW(Database.aisy)
    trs_cxt, _ = ts.profile()

    win = (1000, 2000)
    trs = trs_cxt[0][win[0]:win[1]]
    rel_win, _ = window_cxt(trs_cxt, trs)

    assert_equal(win, rel_win)

    trs2 = trs_cxt[1][win[0]:win[1]]
    with assert_raises(IndexError):
        window_cxt(trs_cxt, trs2)

