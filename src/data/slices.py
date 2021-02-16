import numpy as np


def tvla_slice(traces: np.array, size=-1):
    if size < 0:
        size = len(traces)

    select = np.array(range(size))
    np.random.shuffle(select)

    return np.array([traces[s] for s in np.array_split(select, 4)])


def random_slice(traces, num_slices, even_slices=True):
    """
    Randomly slices up a given NumPy array.
    """
    total = len(traces)
    if even_slices:
        total -= total % num_slices

    indexes = list(range(total))
    np.random.shuffle(indexes)
    ixs_sliced = np.array_split(indexes, num_slices)

    return np.array([traces[s] for s in ixs_sliced])


def random_select(traces, num_traces: int = None):
    """
    Randomly selects a given number of traces from the trace set.
    Shuffles the trace set in the process.
    """
    indexes = list(range(len(traces)))
    np.random.shuffle(indexes)

    return traces[indexes][:num_traces]
