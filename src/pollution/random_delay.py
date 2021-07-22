import numpy as np
from numpy import random
from tqdm import tqdm


def random_delay(traces: np.ndarray, a: int, b: int, delay_amplitude: int, delay_probability=.5):
    """
    Based on the implementation of L. Wu & S. Picek (2020): "Remove Some Noise: On Pre-processing of Side-channel
        Measurements with Autoencoders."
    """
    res = np.zeros_like(traces)
    num_traces, trace_length = traces.shape
    max_sp_value = np.iinfo(np.int8).max

    assert a >= b

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

                    spike = min(trace[sp_old] + delay_amplitude, max_sp_value)
                    res[ix, sp_new:sp_new + 3] = [trace[sp_old], spike, trace[sp_old + 1]]
                    sp_new += 3

    return res
