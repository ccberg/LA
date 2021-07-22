import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import show
from pandas import DataFrame as Df

from src.tools.file import get_plot_path


def store_sns(g, image_name: str):
    """
    Stores a plot figure, if a file name is given.
    File name should be without file extension.
    """
    if image_name is not None:
        img_path = get_plot_path(image_name)
        g.figure.savefig(img_path, format="svg")


def init_plots():
    """
    Sets the correct plot settings for all plots in a notebook.
    """
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 300


def shadow_plot(lines: dict, **args):
    """
    Plots 2 lines, the second one as a "shadow" line over the first.
    """
    keys = list(lines.keys())
    assert len(keys) == 2

    sns.lineplot(data=Df({keys[0]: lines[keys[0]]}))

    line2_args = {"data": Df({keys[1]: lines[keys[1]]}), "palette": [sns.color_palette()[1]], "alpha": .5}
    if "split_y" in args:
        if args["split_y"]:
            line2_args["ax"] = plt.twinx()

        del args["split_y"]

    sns.lineplot(**line2_args).set(**args)

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
    highlight = sns.color_palette()[3]

    for a, b in poi:
        ax.axvspan(a, b, alpha=poi_alpha, color=highlight)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(facecolor=highlight, alpha=poi_alpha))
    labels.append("Point of Interest")

    ax.legend(handles=handles, labels=labels)

    plt.show()

    show(block=False)


def plot_longform(traces):
    sns.set_style('whitegrid')

    longform = []
    for trace in traces:
        for ix in range(len(trace)):
            longform.append((ix, trace[ix]))

    # TODO refine
    cols = ["Sample point", "Power"]
    df = Df(longform, columns=cols)
    sns.lineplot(data=df, x=cols[0], y=cols[1])


def plot_accu(accu: dict, title: str = "", sub_sample=False):
    """
    Plots p-gradients from a given set of accumulators.
    """
    # def get_sparse(gradient):
    #     if sub_sample:
    #         return np.arange(0, len(gradient), len(gradient) / 100).astype(int)
    #     return gradient
    min_len = min([len(ls.p_gradient) for ls in list(zip(*accu.items()))[1]])
    plot_p_gradient(dict([(n, np.array(a.p_gradient[:min_len])) for n, a in accu.items()]), title)


PALETTE_GRADIENT = "mako"


def plot_p_gradient(gradients: dict, title: str = "", max_traces: int = None, min_y: float = 10 ** -32,
                    palette=None, file_name=None):
    """
    Plots p-gradients. Supply a file name if the file should be stored.
    """
    init_plots()

    max_len = 0
    for k in gradients:
        gradients[k] = gradients[k][:max_traces]
        max_len = max(max_len, len(gradients[k]))

    g = sns.lineplot(data=gradients, palette=palette)
    sns.lineplot(data={"Threshold": np.ones(max_len) * 10 ** -5},
                 palette=["red"], dashes=[(2, 2)])

    g.set(yscale="log", ylabel="$p$-value for dist. $A \\neq$ dist. $B$", xlabel="Number of attack traces",
          title=title, ylim=(min_y, 1))
    g.invert_yaxis()

    show(block=False)
    store_sns(g, file_name)
