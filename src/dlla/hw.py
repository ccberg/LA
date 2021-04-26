import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import stats
from tensorflow.python.keras import Model
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.data.ascad import TraceGroup

NUM_CLASSES = 9  # Byte ranges from 00 (HW = 0) to FF (HW = 8), resulting in 9 classes for HW.


def encode(y):
    """
    One-hot encode labels
    """
    return to_categorical(y, num_classes=NUM_CLASSES)


def prepare_traces(profile, attack):
    """
    Normalizes the traces, one-hot encodes the labels.
    Returns profiling traces, labels and attack traces, labels.
    """

    # Normalize traces
    prof_mean, prof_std = profile[0].mean(axis=0), profile[0].std(axis=0)
    x_prof = (profile[0] - prof_mean) / prof_std
    x_att = (attack[0] - prof_mean) / prof_std

    return x_prof, encode(profile[1]), x_att, encode(attack[1])


def fetch_traces(tg: TraceGroup):
    profile = tg.profile.traces, tg.profile.hw_labels()
    attack = tg.attack.traces, tg.attack.hw_labels()

    return prepare_traces(profile, attack)


def hamming_weight_prediction(mdl: Model, x: np.array, num_rows=None):
    """
    Uses the weighted mean of the prediction to predict the hamming weight as a real number between [0..8].
    """
    pred = mdl.predict(x)[:num_rows]
    return np.sum(pred * range(9), axis=1)


def split_by_hw(mdl: Model, x_attack: np.array, y_attack: np.array, splits=1):
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

    return np.array_split(l4, splits), np.array_split(g4, splits)


def get_p_values(a1: np.array, a2: np.array, b2: np.array, test=stats.ttest_ind):
    """
    Calculates the outer (A vs. B) and inner (A vs. A) p-values for both sets having the same distribution.
    """
    pv_ab = test(a1, b2, equal_var=False)[1]
    pv_aa = test(a1, a2, equal_var=False)[1]

    return pv_ab, pv_aa


def dlla_hw(mdl: Model, x_attack: np.array, y_attack: np.array):
    """
    Categorizes traces into two classes: Class A: the hamming weight label is below 4, and class B: above 4.
    Calculates the outer (A vs. B) and inner (A vs. A) p-values for both sets having the same distribution.
    """
    (l4a, l4b), (_, g4b) = split_by_hw(mdl, x_attack, y_attack, 2)
    return get_p_values(l4a, l4b, g4b, stats.ttest_ind)


def p_gradient_dl_la(mdl: Model, x_attack: np.array, y_attack: np.array, max_traces: int):
    """
    Creates a gradient of the p-value that predictions for A and B follow the same distribution.
    Class A consists of traces which are labelled with a hamming weight below 4 and class B above 4.

    Calculates 100 steps between 2 traces and a provided maximum number of traces.
    """
    (l4a, l4b), (_, g4b) = split_by_hw(mdl, x_attack, y_attack, 2)

    gradient, inner_gradient = [], []

    nr = min(len(l4b), len(g4b), max_traces)
    ixs = np.linspace(2, nr, 100).astype(int)

    for i in ixs:
        pv_ab, pv_aa = get_p_values(l4a[:i], l4b[:i], g4b[:i], stats.ttest_ind)

        gradient.append(pv_ab)
        inner_gradient.append(pv_aa)

    df = pd.DataFrame({"A vs. B": gradient, "A vs. A": inner_gradient})
    df = df.set_index(ixs, drop=True)

    return df


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


def plot_gradient(mdl: Model, x_attack: np.array, y_attack: np.array, max_traces=500):
    """
    Plots the min-p gradient for the probability that predictions for classes
    A (HW < 4) and B (HW > 4) are of the same distribution.
    """
    g = sns.lineplot(data=p_gradient_dl_la(mdl, x_attack, y_attack, max_traces))
    g.set(yscale="log", xlabel="Samples", ylabel="$p$-value",
          title="Attack trace predictions, $p$-gradient.\n A: $HW < 4$. B: $HW > 4$")
    g.invert_yaxis()
    g.axhline(10 ** -5, ls='--', color="red")
