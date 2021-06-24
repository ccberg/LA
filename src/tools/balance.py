import numpy as np


def balance(x, y):
    """
    Balances a 2-class labelled trace set.
    """
    y_num = np.argmax(y, axis=1).astype(bool)

    imbalance = len(y) - 2 * np.sum(y_num)
    if imbalance < 0:
        ixs = np.where(y_num)[0]
    else:
        ixs = np.where(~y_num)[0]

    np.random.shuffle(ixs)
    drop_ixs = ixs[:abs(imbalance)]

    return np.delete(x, drop_ixs, axis=0), np.delete(y, drop_ixs, axis=0)
