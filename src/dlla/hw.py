import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import stats
from tensorflow.python.keras import Model
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.data.ascad import TraceGroup

NUM_CLASSES = 9  # Byte ranges from 00 (HW = 0) to FF (HW = 8), resulting in 9 classes for HW.


def encode(y, num_classes=NUM_CLASSES):
    """
    One-hot encode labels
    """
    return to_categorical(y, num_classes=num_classes)


def prepare_traces_dl(x, y, x_att, y_att):
    """
    Normalizes the traces, one-hot encodes the labels.
    Returns profiling traces, labels and attack traces, labels.
    """
    prof_mean, prof_std = x.mean(axis=0), x.std(axis=0)
    norm_x = (x - prof_mean) / prof_std
    norm_x_att = (x_att - prof_mean) / prof_std

    return norm_x, encode(y), norm_x_att, encode(y_att)


def fetch_traces(tg: TraceGroup):
    profile, attack = tg.profile, tg.attack

    return prepare_traces_dl(profile.traces, profile.hw_labels(), attack.traces, attack.hw_labels())


def hamming_weight_prediction(mdl: Model, x: np.array, num_rows=None):
    """
    Uses the weighted mean of the prediction to predict the hamming weight as a real number between [0..8].
    """
    pred = mdl.predict(x)[:num_rows]
    return np.sum(pred * range(9), axis=1)


def split_by_hw(mdl: Model, x_attack: np.array, y_attack: np.array):
    """
    Split traces in two classes.
    Class A: the hamming weight label is below 4, and class B: above 4.

    Returns traces for classes A and B
    """
    hws = np.argmax(y_attack, axis=1)
    nr = min(len(x_attack), len(y_attack))

    l4 = hamming_weight_prediction(mdl, x_attack[np.where(hws < 4)], nr)
    g4 = hamming_weight_prediction(mdl, x_attack[np.where(hws > 4)], nr)

    np.random.shuffle(l4), np.random.shuffle(g4)

    return l4, g4


def get_p_values(low: np.array, high: np.array, test=stats.ttest_ind):
    return test(low, high, equal_var=False)[1]


def dlla_hw(mdl: Model, x_attack: np.array, y_attack: np.array):
    """
    Categorizes traces into two classes: Class A: the hamming weight label is below 4, and class B: above 4.
    Calculates the outer (A vs. B) and inner (A vs. A) p-values for both sets having the same distribution.
    """
    low, high = split_by_hw(mdl, x_attack, y_attack)
    return get_p_values(low, high, stats.ttest_ind)


def dlla_p_gradient(mdl: Model, x_attack: np.array, y_attack: np.array):
    """
    Creates a gradient of the p-value that predictions for A and B follow the same distribution.
    Class A consists of traces which are labelled with a hamming weight below 4 and class B above 4.

    Calculates 100 steps between 2 traces and a provided maximum number of traces.
    """
    low, high = split_by_hw(mdl, x_attack, y_attack)

    nr = min(len(low), len(high))
    gradient = np.ones(nr)

    for i in range(nr):
        gradient[i] = get_p_values(low[:i], high[:i])

    # Every t-test p-value consumes 2 traces.
    return np.repeat(gradient, 2)


def plot_predictions(mdl: Model, x_attack: np.array, y_attack: np.array):
    """
    Plots the distribution of the predictions of the provided model.
    Separated in two classes (HW < 4, HW > 4)
    """
    less_4, greater_4 = split_by_hw(mdl, x_attack, y_attack)
    num_all = len(less_4[0]) + len(greater_4[0])

    g = sns.histplot(data={
        "HW < 4": less_4[0],
        "HW > 4": greater_4[0]
    }, bins=80, binrange=(0, 8))
    g.set(xlabel="Predicted hamming weight, bin size = 0.1",
          title=f"Attack trace predictions for {num_all} sample traces.")

    return g


def plot_gradient(mdl: Model, x_attack: np.array, y_attack: np.array, max_traces=500):
    """
    Plots the min-p gradient for the probability that predictions for classes
    A (HW < 4) and B (HW > 4) are of the same distribution.
    """
    g = sns.lineplot(data=dlla_p_gradient(mdl, x_attack, y_attack, max_traces))
    g.set(yscale="log", xlabel="Samples", ylabel="$p$-value",
          title="Attack trace predictions, $p$-gradient.\n A: $HW < 4$. B: $HW > 4$")
    g.invert_yaxis()
    g.axhline(10 ** -5, ls='--', color="red")


def split_traces(x, y, balance=False):
    yam = np.argmax(y, axis=1)
    a = x[np.where(yam < 4)]
    b = x[np.where(yam > 4)]

    limit = None
    if balance:
        limit = min(len(a), len(b))

    return a[:limit], b[:limit]
