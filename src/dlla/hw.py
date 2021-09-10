import numpy as np
import seaborn as sns
from scipy.stats import stats
from tensorflow.python.keras import Model

from src.data.ascad import TraceGroup
from src.dlla.berg import make_mlp
from src.tools.dl import encode
from src.trace_set.database import Database
from src.trace_set.set_hw import TraceSetHW

NUM_CLASSES = 9  # Byte ranges from 00 (HW = 0) to FF (HW = 8), resulting in 9 classes for HW.


def prepare_traces_dl(x, y, x_att, y_att, num_classes=NUM_CLASSES):
    """
    Normalizes the traces, one-hot encodes the labels.
    Returns profiling traces, labels and attack traces, labels.
    """
    prof_mean, prof_std = x.mean(axis=0), x.std(axis=0)
    norm_x = (x - prof_mean) / prof_std
    norm_x_att = (x_att - prof_mean) / prof_std

    return norm_x, encode(y, num_classes), norm_x_att, encode(y_att, num_classes)


def fetch_traces(tg: TraceGroup):
    profile, attack = tg.profile, tg.attack

    return prepare_traces_dl(profile.traces, profile.hw_labels(), attack.traces, attack.hw_labels())


def predict(mdl: Model, x: np.array, num_classes=NUM_CLASSES):
    """
    Uses the weighted mean of the prediction to predict the hamming weight as a real number between [0..8].
    """
    return np.sum(mdl.predict(x) * range(num_classes), axis=1)


def split_by_hw(mdl: Model, x_attack: np.array, y_attack: np.array):
    """
    Split traces in two classes.
    Class A: the hamming weight label is below 4, and class B: above 4.

    Returns traces for classes A and B
    """
    hws = np.argmax(y_attack, axis=1)

    l4 = predict(mdl, x_attack[np.where(hws < 4)], NUM_CLASSES)
    g4 = predict(mdl, x_attack[np.where(hws > 4)], NUM_CLASSES)

    np.random.shuffle(l4), np.random.shuffle(g4)

    return l4, g4


def prediction_test(low: np.array, high: np.array, test=stats.ttest_ind):
    _, p = test(low, high, equal_var=False)

    if np.isnan(p):
        return 1.

    return p


def dlla_closed_p(mdl: Model, x: np.array, y: np.array):
    """
    p-value for closed-source DL-LA
    """
    la_bit = np.argmax(y, axis=1)
    low, high = predict(mdl, x[~la_bit], 2), predict(mdl, x[la_bit], 2)
    return prediction_test(low, high, stats.ttest_ind)


def dlla_known_p(mdl: Model, x: np.array, y: np.array):
    """
    p-value for open-source DL-LA
    """
    low, high = split_by_hw(mdl, x, y)
    return prediction_test(low, high, stats.ttest_ind)


def dlla_p_gradient(mdl: Model, x: np.array, y: np.array):
    """
    Creates a gradient of the p-value that predictions for A and B follow the same distribution.
    Class A consists of traces which are labelled with a hamming weight below 4 and class B above 4.
    """
    low, high = split_by_hw(mdl, x, y)

    nr = min(len(low), len(high))
    gradient = np.ones(nr)

    for i in range(nr):
        gradient[i] = prediction_test(low[:i], high[:i])

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


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.list_physical_devices("GPU")
    while True:
        trace_set = TraceSetHW(Database.ascad, limits=(None, 1000))
        x9, y9, x9_att, y9_att = prepare_traces_dl(*trace_set.profile(), *trace_set.attack(), num_classes=9)
        np.random.shuffle(y9_att)

        mdl_unknown = make_mlp(x9, y9, num_classes=9)
        dlla_k_p = dlla_known_p(mdl_unknown, x9_att, y9_att)

        print("9 class", dlla_k_p)

        # trace_set = TraceSetHW(Database.ascad, Pollution(PollutionType.desync, 1), limits=(None, 1000))
        # x2, y2, x2_att, y2_att = prepare_traces_dl(*trace_set.profile_la(), *trace_set.attack_la(), num_classes=2)
        # np.random.shuffle(y2)
        # np.random.shuffle(y2_att)
        #
        # mdl_unknown = make_mlp(x2, y2, progress=False, num_classes=2)
        # dlla_uk_p = dlla_closed_p(mdl_unknown, x2_att, y2_att)
        #
        # print("2 class", dlla_uk_p)
