import numpy as np

from src.trace_set.pollution import PollutionType


def max_data(traces: np.ndarray):
    return np.iinfo(traces.dtype).max


def windowed(raw_traces: np.ndarray, window: (int, int), calculate_win=2):
    wa, wb = window
    win_size = wb - wa

    traces = raw_traces[:, wa:wb + win_size * calculate_win]
    return traces, win_size


def gen_test_data():
    raw_traces = np.array([(np.sin(np.linspace(1, 20, 2000)) * 100) for _ in range(100)], dtype=np.int8)
    window = (500, 1500)
    traces = raw_traces[:, window[0]:window[1]]

    return raw_traces, window, traces


def file_suffix(poll_type: PollutionType, poll_param):
    suffix = ""
    if poll_type is not None:
        suffix = f"-{poll_type.name}-{poll_param}"

    return suffix
