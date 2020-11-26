import os
import numpy as np

from tools.constants import ROOT_DIR


def cache_np(name, f=None, *args, v=1, replace=False):
    """
    Caches the result of a function. In case no function is supplied,
    the results of the specified cache name are returned.
    """
    f_name = f"{ROOT_DIR}/cache/{name}_{v}.npz"
    if f is not None:
        # Only try to store the results of f(*args) if f is supplied.
        if replace or not os.path.exists(f_name):
            np.savez_compressed(f_name, data=f(*args))

    with np.load(f_name, allow_pickle=True) as file:
        return file['data']


if __name__ == '__main__':
    print(cache_np("test", lambda a: np.zeros(a), 100))

