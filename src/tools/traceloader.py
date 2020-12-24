import numpy as np
from collections import Counter

from src.data import ASCADData, ASCADDataType
import pandas as pd


class TraceCategory:
    def __init__(self, trace_category, trace_range):
        self.t_range = trace_range

        self.traces = np.array(trace_category["traces"])
        self.labels = np.array(trace_category["labels"])

        self.tk_cache = {}
        self.ct_cache = {}

    def filter_traces(self, label):
        if label not in self.tk_cache:
            # TODO Warning here
            ixs = np.where(np.array(self.labels) == label)[0]
            self.tk_cache[label] = np.array(self.traces[ixs])

        return self.tk_cache[label]

    def contingency_table(self, label):
        """
        Builds a contingency table from traces from the dataset for a given label.

        :param label: the label for which traces the contingency table will be build.
        :return: the contingency table as a numpy array.
        """
        if label not in self.ct_cache:
            df = pd.DataFrame([Counter(bins) for bins in self.filter_traces(label)])
            res = df.sum().sort_index().reindex(self.t_range, fill_value=0).values
            self.ct_cache[label] = np.array(res, dtype=int)

        return self.ct_cache[label]


class TraceGroup:
    def __init__(self, trace_group, trace_range, category_type=TraceCategory):
        self.profile = category_type(trace_group["Profiling_traces"], trace_range)
        self.attack = category_type(trace_group["Attack_traces"], trace_range)


class RandomTCat(TraceCategory):
    def __init__(self, trace_category, trace_range):
        super().__init__(trace_category, trace_range)

        np.random.shuffle(self.labels)


class MaskedKeyTCat(TraceCategory):
    def __init__(self, trace_category, trace_range):
        super().__init__(trace_category, trace_range)

        self.labels = trace_category["labels_mask"]


class ASCAD:
    key_size = 256
    trace_len = 1400
    offset = -128

    def __init__(self):
        self.default = TraceGroup(ASCADData.random_key(), ASCADData.data_range)
        self.random = TraceGroup(ASCADData.random_key(), ASCADData.data_range, RandomTCat)
        self.masked = TraceGroup(ASCADData.random_key(), ASCADData.data_range, MaskedKeyTCat)
        self.desync_50 = TraceGroup(ASCADData.random_key(ASCADDataType.desync_50), ASCADData.data_range)
        self.desync_100 = TraceGroup(ASCADData.random_key(ASCADDataType.desync_50), ASCADData.data_range)
