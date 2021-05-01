import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.tools.lists import concat


def z_norm(x_profiling, x_attack):
    profiling_mean, profiling_var = x_profiling.mean(axis=0), x_profiling.var(axis=0)

    def normalize(x):
        return (x - profiling_mean) / profiling_var

    return normalize(x_profiling), normalize(x_attack)


def labelize(class_traces):
    """
    Labels the provided traces.
    class_traces should be an array of trace groups. Each group gets labelled with it's index.
    So the first group gets label 0, the second label 1, etc.

    Returns the Concatenates the traces and one
    """
    num_classes = len(class_traces)

    res = []
    for cls, traces in enumerate(class_traces):
        res.append([cls] * len(traces))

    return concat(class_traces), concat(res), num_classes


# Percentage of traces that will go in the attack trace set.
ATTACK_RATIO = .2


def prepare_dlla(traces, labels, num_classes):
    """
    Prepares traces for usage in DL-LA training.
    """
    # Randomize traces and labels
    indices = np.arange(len(traces))
    np.random.shuffle(indices)
    x, y = traces[indices], labels[indices]

    # Separate attack from profiling traces
    num_profiling = round(1 - ATTACK_RATIO * len(traces))
    x_profiling, x_attack = x[:num_profiling], x[num_profiling:]
    y_profiling, y_attack = y[:num_profiling], y[num_profiling:]

    # Normalize traces
    x_profiling, x_attack = z_norm(x_profiling, x_attack)

    # One-hot encode labels
    y_profiling, y_attack = to_categorical(y_profiling, num_classes), to_categorical(y_attack, num_classes)

    return x_profiling, y_profiling, x_attack, y_attack
