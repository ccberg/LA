import numpy as np
import pandas as pd


# TODO try window = 100, window bug.
def select_poi(diff, window=15):
    """
    Selects the top n most interesting points of interest,
    based on some statistical moment difference.
    """
    std = np.std(diff)
    # We are only interested in large distances between sample points in non-equal keys and
    #   small distances in sample points between equal keys.
    ts = [*(diff > std), *([False] * window)]

    acc = []
    counter, sum_diff, max_diff = 0, 0.0, 0.0
    for ix in range(len(ts)):
        if ts[ix]:
            counter += 1
            sum_diff += diff[ix]
            max_diff = max(diff[ix], max_diff)

        elif counter >= window:
            avg_diff = sum_diff / counter

            if avg_diff > 2 * std:
                acc.append((ix - counter, ix, avg_diff))

            counter, sum_diff, max_diff = 0, 0.0, 0.0

    if len(acc) == 0:
        return []

    return pd.DataFrame(acc).sort_values(2, ascending=False)[[0, 1]].values
