import numpy as np
from numpy import random
from tqdm import tqdm

from src.pollution.tools import max_data


def random_delay(traces: np.ndarray, a: int, b: int, delay_amplitude: int, delay_probability=.5):
    """
    Based on the implementation of L. Wu & S. Picek (2020): "Remove Some Noise: On Pre-processing of Side-channel
        Measurements with Autoencoders."
    """
    res = np.zeros_like(traces)
    num_traces, trace_length = traces.shape

    max_sp = np.max(traces)
    norm_factor = max_data(traces) / (max_sp + delay_amplitude)
    if norm_factor < 1:
        traces = np.array(norm_factor * traces, dtype=traces.dtype)
        delay_amplitude *= norm_factor

    for ix, trace in tqdm(enumerate(traces), total=num_traces, desc=f"Random delay ({delay_probability})"):
        sp_old, sp_new = 0, 0

        # Computing (too much) random variables all at once yields >2x speed increase.
        do_jitter = random.binomial(1, delay_probability, trace_length)
        lower_bound = random.randint(0, a - b, size=trace_length)
        upper_bound = random.randint(0, b, size=trace_length) + lower_bound

        while sp_new < trace_length and sp_old < trace_length:
            r = do_jitter[sp_new]

            res[ix, sp_new] = trace[sp_old]
            sp_old += 1
            sp_new += 1

            if r:
                for _ in range(upper_bound[sp_old]):
                    if sp_new + 3 > trace_length:
                        continue

                    spike = trace[sp_old] + delay_amplitude
                    delay_sequence = [trace[sp_old], spike, trace[sp_old + 1]]
                    res[ix, sp_new:sp_new + 3] = delay_sequence
                    sp_new += 3

    return res
