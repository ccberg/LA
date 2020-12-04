import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame as Df
from matplotlib.pyplot import show


def shadow_plot(lines: dict, **args):
    """
    Plots 2 lines, the second one as a "shadow" line over the first.
    """
    keys = list(lines.keys())
    assert len(keys) == 2

    sns.lineplot(data=Df({keys[0]: lines[keys[0]]}))
    sns.lineplot(data=Df({keys[1]: lines[keys[1]]}), palette=[sns.color_palette()[1]], alpha=.5).set(**args)

    show(block=False)


def line_plot(lines: dict, **args):
    """
    Your average line plot.
    """
    sns.lineplot(data=Df(lines)).set(**args)

    show(block=False)


def line_plot_poi(lines: dict, poi: list, poi_alpha=.3, **args):
    """
    Plots up to 3 lines with points of interest, shaded red.
    """
    assert len(lines) < 3  # For visibility.

    fig, ax = plt.subplots()
    sns.lineplot(data=Df(lines)).set(**args)

    for a, b in poi:
        ax.axvspan(a, b, alpha=poi_alpha, color=sns.color_palette()[3])

    plt.show()

    show(block=False)
