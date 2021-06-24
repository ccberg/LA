from collections.abc import Iterator

import numpy as np
from tqdm import tqdm


class Jitterator(Iterator):
    """
    Iterator for Jitter.
    """

    def __init__(self, trace, exp):
        super()

        trace_length = len(trace)
        jitter_lengths = np.random.exponential(exp, trace_length).astype(int)
        jitter_ixs = np.where(jitter_lengths > 0)

        self.splits = iter(np.split(trace, jitter_ixs[0] + 1))
        self.jitters = zip(trace[jitter_ixs], jitter_lengths[jitter_ixs])

    def __next__(self):
        elem, jl = next(self.jitters)
        return next(self.splits), np.array([elem] * jl)


def jitter_trace(trace, exp=1):
    """
    Simulates jitter by randomly and independently duplicating sample points.
    Random shift is modelled by an exponential distribution with a given value for the rate parameter.
    """
    trace_len = len(trace)
    jitterator = Jitterator(trace, exp)

    res = []
    while len(res) < trace_len:
        s, j = next(jitterator)
        res.extend(s)
        res.extend(j)

    return np.array(res[:len(trace)])


def jitter(traces, exp=1):
    """
    Simulates jitter using the given rate parameter. Applies it to the supplied traces.
    """
    res = np.zeros_like(traces)

    for ix in tqdm(range(len(traces)), desc=f"Applying jitter with exp={exp}"):
        res[ix] = jitter_trace(traces[ix], exp)

    return res


if __name__ == '__main__':
    jitter_trace(np.arange(10))
