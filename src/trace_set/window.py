import numpy as np
from tqdm import tqdm

from src.tools.lists import find


def get_windows(raw_traces, sample_trace):
    """
    Calculates the position of a context window for trace pollution such as clock jitter.

    Returns the window of the actual point of interest relative to the context window,
        as well as the context window itself.
    """
    start_ix = find(raw_traces[0], sample_trace)

    win_size = len(sample_trace)
    window = np.array((start_ix, start_ix + win_size))

    window_margin = win_size

    # Window with context
    window_cxt = window + np.array((-window_margin, window_margin))

    # Return the window relative to the window context.
    rel_window = window - window_cxt[0]

    return rel_window, window_cxt


def extract_traces(raw_traces, window_cxt):
    """
    Extracts traces based on some provided window.

    Returns only the parts of the trace within that window.
    """
    num_traces, _ = raw_traces.shape

    wa, wb = window_cxt
    win_size = wb - wa

    traces_cxt = np.zeros((num_traces, win_size), dtype=np.int8)
    for ix in tqdm(range(num_traces)):
        traces_cxt[ix] = raw_traces[ix, window_cxt[0]:window_cxt[1]]

    return traces_cxt
