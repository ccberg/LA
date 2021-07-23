import numpy as np
from numpy import random
from tqdm import tqdm


def window_jitter(raw_traces: np.ndarray, window: (int, int), noise_level: float):
    """
    Based on the implementation of R. Benadjila et al. (2018) "Study of Deep Learning Techniques for Side-Channel
        Analysis and Introduction to ASCAD Database-Long Paper."
    """
    wa, wb = window
    window_length = wb - wa
    num_traces, trace_len = raw_traces.shape
    jitter_start = random.normal(scale=noise_level, size=num_traces).astype(int)

    res = np.zeros((num_traces, window_length), dtype=np.int8)
    for ix, trace in tqdm(enumerate(raw_traces), total=len(raw_traces), desc=f"Window jitter ({noise_level})"):
        offset = jitter_start[ix]
        res[ix] = trace[wa + offset:wb + offset]

    return res
