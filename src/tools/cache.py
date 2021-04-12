import errno
import os
import numpy as np

from src.tools.constants import ROOT_DIR
import pickle as pkl


def get_cache_loc():
    return f"./.cache"


def mkdir_suppress_exist(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def make_dirs(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def cache_np(name, f=None, *args, v=1, replace=False):
    """
    Caches the result of a function. In case no function is supplied,
    the results of the specified ..cache name are returned.
    """
    file_name = f"{get_cache_loc()}/{name}_{v}.npz"

    if f is not None:
        # Only try to store the results of f(*args) if f is supplied.
        if replace or not os.path.exists(file_name):
            make_dirs(file_name)

            np.savez_compressed(file_name, data=f(*args))

    with np.load(file_name, allow_pickle=True) as file:
        return file['data']


def cache_pkl(name, f=None, *args, v=1, replace=False):
    file_name = f"{get_cache_loc()}/{name}_{v}.pkl"

    if f is not None:
        # Only try to store the results of f(*args) if f is supplied.
        if replace or not os.path.exists(file_name):
            make_dirs(file_name)

            with open(file_name, "wb") as file:
                pkl.dump(f(*args), file)

    with open(file_name, "rb") as file:
        return pkl.load(file)


class NBCache:
    def __init__(self, path, version=None):
        self.path = path

        if version is not None:
            self.path += "/" + version

        os.makedirs(get_cache_loc() + "/" + self.path, exist_ok=True)

    def np(self, name: str, *args):
        res = cache_np(f"{self.path}/{name}", *args)
        setattr(self, name, res)

        return res


if __name__ == '__main__':
    print(cache_np("test", lambda a: np.zeros(a), 100))
