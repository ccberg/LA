import os
import numpy as np

from src.tools.constants import ROOT_DIR
import pickle as pkl


def get_cache_loc():
    return f"{ROOT_DIR}/../.cache"


def cache_np(name, f=None, *args, v=1, replace=False):
    """
    Caches the result of a function. In case no function is supplied,
    the results of the specified .cache name are returned.
    """
    f_name = f"{get_cache_loc()}/{name}_{v}.npz"
    if f is not None:
        # Only try to store the results of f(*args) if f is supplied.
        if replace or not os.path.exists(f_name):
            np.savez_compressed(f_name, data=f(*args))

    with np.load(f_name, allow_pickle=True) as file:
        return file['data']


def cache_pkl(name, cls_name=None, *args, v=1, replace=False):
    f_name = f"{get_cache_loc()}/{name}_{v}.pkl"

    if cls_name is not None:
        # Only try to store the results of cls_name(*args) if cls_name is supplied.
        if replace or not os.path.exists(f_name):
            with open(f_name, "wb") as file:
                pkl.dump(cls_name(*args), file)

    with open(f_name, "rb") as file:
        return pkl.load(file)


class NBCache:
    def __init__(self, path):
        self.path = path

    def np(self, name: str, *args):
        res = cache_np(f"{self.path}/{name}", *args)
        setattr(self, name, res)

        return res


if __name__ == '__main__':
    print(cache_np("test", lambda a: np.zeros(a), 100))
