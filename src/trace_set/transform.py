import numpy as np

from src.dlla.hw import prepare_traces_dl
from tensorflow.python.keras.utils.np_utils import to_categorical

# TODO replace with mlp_hw notebook variants
from src.tools.dl import encode
from src.tools.la import balance
from src.trace_set.database import Database
from src.trace_set.pollution import PollutionType, Pollution
from src.trace_set.set_hw import TraceSetHW


def reduce_fixed_fixed(x, y, balanced=False):
    """
    Takes 9-class (categorical) hamming weight labels and reduces it to 2 semi-fixed classes.
    """
    hamming_weight = np.argmax(y, axis=1)
    filter_ixs = hamming_weight != 4
    is_high = hamming_weight[filter_ixs] > 4

    traces, la_bit = x[filter_ixs], is_high

    if balanced:
        traces, la_bit = balance(traces, la_bit)

    return traces, encode(la_bit, 2)


def reduce_fixed_random(x, y, balanced=False):
    """
    Takes 9-class (categorical) hamming weight labels and reduces it to 2 classes: semi-fixed and random.
    """
    hamming_weight = np.argmax(y, axis=1)
    is_random = np.random.binomial(1, .5, len(x)).astype(bool)
    filter_ixs = np.logical_or(hamming_weight < 4, is_random)

    traces, la_bit = x[filter_ixs], is_random[filter_ixs]
    if balanced:
        traces, la_bit = balance(traces, la_bit)

    return traces, encode(la_bit, 2)


if __name__ == '__main__':
    trace_set = TraceSetHW(Database.ascad, Pollution(PollutionType.gauss, 0), limits=(1000, 1000))
    x9, y9, x9_att, y9_att = prepare_traces_dl(*trace_set.profile(), *trace_set.attack())
    x2, y2 = reduce_fixed_fixed(x9, y9, balanced=True)

    print(x2)