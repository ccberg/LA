import os
import numpy as np

from tools.constants import ROOT_DIR


def cache_np(name, f, *args, v=1, replace=False):
    f_name = f"{ROOT_DIR}/cache/{name}_{v}.npz"
    if replace or not os.path.exists(f_name):
        np.savez_compressed(f_name, data=f(*args))

    with np.load(f_name) as file:
        return file['data']


if __name__ == '__main__':
    print(cache_np("test", lambda a: np.zeros(a), 100))

