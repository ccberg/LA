import numpy as np
from numpy import random
from tqdm import tqdm

from src.pollution.tools import windowed


def clock_jitter(raw_traces: np.ndarray, window: (int, int), clock_var: float):
    """
    Based on the implementation of L. Wu & S. Picek (2020): "Remove Some Noise: On Pre-processing of Side-channel
        Measurements with Autoencoders."
    """
    traces, win_size = windowed(raw_traces, window)
    num_traces, trace_length = traces.shape

    res = np.zeros_like(traces)
    min_trace_len = trace_length

    for ix, trace in tqdm(enumerate(traces), total=num_traces, desc=f"Clock jitter ({clock_var})"):
        sp_old, sp_new = 0, 0

        # Computing (too much) random variables all at once yields a ~1.5x speed increase.
        jitters = random.normal(scale=clock_var, size=trace_length).astype(int)

        while sp_new < trace_length and sp_old < trace_length:
            r = jitters[sp_new]

            res[ix, sp_new] = trace[sp_old]
            sp_old += 1
            sp_new += 1

            sp_old_r = sp_old - r
            sp_new_r = sp_new + r
            if sp_old_r > trace_length or sp_new_r > trace_length or sp_old + 1 >= trace_length:
                break

            # if r < 0, delete r point afterward
            if r <= 0:
                sp_old = sp_old_r
            # if r > 0, add r point afterward
            else:
                avg_point = (int(trace[sp_old]) + int(trace[sp_old + 1])) / 2
                res[ix, sp_new:sp_new_r] = avg_point
                sp_new = sp_new_r

        min_trace_len = min(sp_new, min_trace_len)

    return res[:, :win_size]
