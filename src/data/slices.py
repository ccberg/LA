import numpy as np


def make_slice(traces: np.array, size=-1):
    if size < 0:
        size = len(traces)

    select = np.array(range(size))
    np.random.shuffle(select)

    return np.array([traces[s] for s in np.array_split(select, 4)])
