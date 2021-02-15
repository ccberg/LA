import math
import os

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.data.smote import smote
from src.data.traceloader import ASCAD
from src.tools.cache import get_cache_loc, NBCache
from src.data.traceloader import TraceCategory


def ctable_mv(ctable: np.array, num_observations: int):
    """
    Calculates the mean and variance from a contingency table with a corresponding number of observations.
    """
    # Using 128-bit floats prevents some rounding errors when comparing with the np implementation of var and mean.
    ixs = np.arange(0, len(ctable), dtype=np.float128)

    mu = (ctable * ixs).sum() / num_observations
    sigma2 = ((ctable * ixs ** 2).sum() / num_observations) - (mu ** 2)

    return mu, sigma2


class CTableStore:
    def __init__(self, tc: TraceCategory, pref_size: int, key_range=256, offset=128):
        self.tc = tc
        self.key_range = key_range
        self.offset = offset

        max_slice_size = max([math.ceil(len(self.tc.filter_by_key(k)) / 4) for k in range(self.key_range)])
        self.size = max(pref_size, max_slice_size)

        t_max = max([self.tc.filter_by_key(k).max() for k in range(self.key_range)]) + self.offset
        t_min = min([self.tc.filter_by_key(k).min() for k in range(self.key_range)]) + self.offset

        self.trace_type = np.uint8
        # Maximum value of trace should be lower than the capacity of uint8.
        assert t_min > 0
        assert t_max < np.iinfo(self.trace_type).max

        self.table_type = np.uint16
        # Maximum value of bincount should be lower than the capacity of uint16.
        assert self.size < np.iinfo(self.table_type).max

        self.slices = self.gen_slices()
        self.smoted = self.apply_smote(self.slices)
        self.tables = self.gen_tables(self.smoted)

    def gen_slices(self):
        """
        Slices the data up in two internally disjoint data sets (A and B) per key byte.
        The set A contains 4 slices, the set B contains 2 slices.

        B is generated by getting 4 random slices and discarding 2 of them. This prevents the slices in A from
        being oversampled more than twice as often as the slices in B in the SMOTEing phase.
        """
        res = {}

        for k in tqdm(range(self.key_range), "Slicing up traces"):
            ts = self.tc.filter_by_key(k) + self.offset
            uts = np.array([np.array(t, dtype=self.trace_type) for t in ts])

            np.random.shuffle(uts)
            ts4 = np.array(np.array_split(uts.copy(), 4))
            np.random.shuffle(uts)
            ts2 = np.array(np.array_split(uts, 4))

            res[k] = np.array([ts4, ts2[:2]])

        return res

    def apply_smote(self, slices):
        """
        Applies SMOTE to the slices, resulting in all slices from all sets containing the same amount of samples.
        """
        res = {}

        for k in tqdm(range(self.key_range), "SMOTEing traces"):
            res[k] = np.array(smote(slices[k], self.size, self.trace_type))

        return res

    def gen_tables(self, smoted):
        """
        Generates contingency tables by counting the occurrence of each sample point value within a slice,
        for each sample point.
        """
        res = {}

        for k in tqdm(range(self.key_range), "Generating contingency tables"):
            res[k] = {}
            for c, category in enumerate(smoted[k]):
                res[k][c] = {}
                for s, trace_slice in enumerate(category):
                    ts = np.moveaxis(trace_slice, 0, -1)
                    res[k][c][s] = np.array(
                        [np.array(np.bincount(t, minlength=256), dtype=self.table_type) for t in ts])

        return res


def cache_cts(c: NBCache, name: str, t_cat: TraceCategory, pref_size: int):
    fname = f'{get_cache_loc()}/{c.path}/{name}.hdf5'

    if not os.path.exists(fname):
        f = h5py.File(fname, 'a')

        ct = CTableStore(t_cat, pref_size)
        for k, key in tqdm(ct.tables.items(), "Storing tables"):
            for c, category in key.items():
                grp = f.create_group(f"key_{str(k).zfill(3)}/cat_{c}")
                for t, trace_slice in category.items():
                    arr = np.array([sp.astype(np.uint16) for sp in trace_slice])
                    grp.create_dataset(f"slice_{t}", data=arr)

        f.close()

    return h5py.File(fname, 'r')


if __name__ == '__main__':
    ascad = ASCAD()
    cts = CTableStore(ascad.masked.profile, 1)

    test_cts_split = np.array([i[0] for i in cts.slices[0][0][0]])
    test_cts_data = np.array([i[0] for i in cts.smoted[0][0][0]])

    sns.lineplot(data=pd.DataFrame(zip(np.bincount(test_cts_split)[-6:], np.bincount(test_cts_data)[-6:]),
                                   columns=["Source", "Smoted result"]))

    assert np.all(cts.tables[0][0][0][0] == np.bincount(test_cts_data, minlength=256))
