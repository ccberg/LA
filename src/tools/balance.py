import numpy as np

from src.dlla.hw import encode


def balance_bit(x, y_bit):
    """
    Balances a trace set labelled by a LA bit.
    """
    imbalance = len(y_bit) - 2 * np.sum(y_bit)
    if imbalance < 0:
        ixs = np.where(y_bit)[0]
    else:
        ixs = np.where(~y_bit)[0]

    np.random.shuffle(ixs)
    drop_ixs = ixs[:abs(imbalance)]

    return np.delete(x, drop_ixs, axis=0), np.delete(y_bit, drop_ixs, axis=0)


def balance(x, y):
    """
    Balances a 2-class categorically labelled trace set.
    """
    y_bit = np.argmax(y, axis=1).astype(bool)
    balanced_x, balanced_y_bit = balance_bit(x, y_bit)

    return balanced_x, encode(balanced_y_bit, 2)
