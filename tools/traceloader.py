import numpy as np
from collections import Counter

from data.loader.ascad import ASCADData, ASCADDataType
import pandas as pd


class ASCAD:
    def __init__(self):
        self.default = TraceGroup(ASCADData.random_key(), ASCADData.data_range)
        self.random = TraceGroup(ASCADData.random_key(), ASCADData.data_range, True)
        self.desync_50 = TraceGroup(ASCADData.random_key(ASCADDataType.desync_50), ASCADData.data_range)
        self.desync_100 = TraceGroup(ASCADData.random_key(ASCADDataType.desync_50), ASCADData.data_range)


class TraceGroup:
    def __init__(self, trace_group, trace_range, shuffle=False):
        self.profile = TraceCategory(trace_group["Profiling_traces"], trace_range, shuffle)
        self.attack = TraceCategory(trace_group["Attack_traces"], trace_range, shuffle)


class TraceCategory:
    def __init__(self, trace_category, trace_range, shuffle=False):
        self.t_range = trace_range

        self.traces = np.array(trace_category["traces"])
        self.labels = np.array(trace_category["labels"])

        if shuffle:
            np.random.shuffle(self.labels)

        self.tk_cache = {}
        self.ct_cache = {}

    def filter_traces(self, label):
        if label not in self.tk_cache:
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
