import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import stats

from src.tools.poi import select_poi
from src.data.ascad import TraceCategory


def statistical_moment(traces: np.array, moment=1):
    """
    Retrieves a statistical moment in a given order for a given set of traces.
    The moment
    """
    if moment == 1:
        return traces.mean(axis=0)
    if moment == 2:
        return traces.var(axis=0)
    if moment == 3:
        return stats.skew(traces, axis=0)
    if moment == 4:
        return stats.kurtosis(traces, axis=0)

    raise Exception("Moment not implemented.")


def calc_moment_difference(left_1, left_2, right, moment=1):
    """
    Calculates the difference in statistical moment between power traces with
    equal keys and power traces with different keys.
    """

    def smt(a):
        return statistical_moment(a, moment)

    dist_neq = abs(smt(left_1) - smt(right))
    dist_eq = abs(smt(left_1) - smt(left_2))

    return dist_neq - dist_eq


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


def get_moment_differences(tc: TraceCategory, trace_size=ASCAD.trace_len, max_moment=3):
    """
    Calculates the difference in statistical moment between power traces with
    equal keys and power traces with different keys, up to a given order of
    statistical moment.
    """
    mdiff = np.zeros((max_moment + 1, trace_size))
    for stat_moment in range(1, max_moment + 1):
        low, high = tc.filter_by_hw(False), tc.filter_by_hw(True)
        low_1, low_2 = random_slice(low, 2)

        mdiff[stat_moment] = calc_moment_difference(low_1, low_2, high)

    return mdiff


def plot_poi(mdiff, moment):
    """
    Plots moment difference with points of interest.
    """
    fig, ax = plt.subplots()

    title = f"Difference in statistical moment ({moment}) between traces with" \
            f"equal and\ntraces with different keys, Points of Interest are highlighted.\n"
    sns.lineplot(data=mdiff[moment]).set_title(title)

    for a, b in select_poi(mdiff[moment]):
        ax.axvspan(a, b, alpha=0.3, color=sns.color_palette()[3])

    plt.show()


def plot_poi_trace(trace, poi):
    """
    Plots power trace with points of interest.
    """
    fig, ax = plt.subplots()

    title = f"Some power trace, Points of Interest from\nstatistical moment (1) are highlighted.\n"
    sns.lineplot(data=trace, palette=[sns.color_palette()[4]]).set_title(title)

    for a, b in poi:
        ax.axvspan(a, b, alpha=0.3, color=sns.color_palette()[3])

    plt.show()


if __name__ == '__main__':
    ascad = ASCAD()
    moment_diff = get_moment_differences(ascad.default.profile)

    plot_poi(moment_diff, 1)
