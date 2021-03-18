from collections import Counter

import numpy as np
import pandas as pd

from src.data.loaders.ascad import ASCADData


class TraceCategory:
    def __init__(self, trace_category, trace_range):
        self.t_range = trace_range

        self.traces = np.array(trace_category["traces"])
        self.labels = np.array(trace_category["labels"])

        self.hamming_weights = np.array(trace_category["hamming_weights"])
        # 3rd state byte after 1st round SBox
        self.aes_r1b3 = np.array(trace_category["aes_r1b3"])

        self.tk_cache = {}
        self.hw_cache = {}
        self.ct_cache = {}

        # Take the Hamming Weight of the third state byte.
        self.hw_target_byte = 2
        # Take Hamming Weight of the state after SBox from the first round.
        self.hw_target_round = 0

        self.hw_target = 0

    def filter_by_key(self, key):
        """
        Filters traces by a given first key byte.
        """
        if key not in self.tk_cache:
            ixs = np.where(np.array(self.labels) == key)[0]
            self.tk_cache[key] = np.array(self.traces[ixs])

        return self.tk_cache[key]

    def filter_by_hw(self, above_median: bool):
        target_key = int(above_median)
        if target_key not in self.hw_cache:
            hws = self.hw_labels()

            if above_median:
                ixs = np.where(hws < 4)[0]
            else:
                ixs = np.where(hws > 4)[0]

            self.hw_cache[target_key] = self.traces[ixs]

        return self.hw_cache[target_key]

    def hw_labels(self):
        """
        Returns the hamming weight labels for the first byte of the first state all traces in this dataset
        """
        return self.hamming_weights[:, self.hw_target_round, self.hw_target_byte]

    def contingency_table(self, label):
        """
        Builds a contingency table from traces from the dataset for a given label.

        :param label: the label for which traces the contingency table will be build.
        :return: the contingency table as a numpy array.
        """
        if label not in self.ct_cache:
            df = pd.DataFrame([Counter(bins) for bins in self.filter_by_key(label)])
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
        np.random.shuffle(self.hamming_weights)


class MaskedKeyTCat(TraceCategory):
    def __init__(self, trace_category, trace_range):
        super().__init__(trace_category, trace_range)

        self.labels = trace_category["labels_mask"]


class TraceLoader:
    key_size, trace_len, offset = [0] * 3


class AscadRandomKey(TraceLoader):
    key_size = 256
    trace_len = 1400
    offset = -128

    def __init__(self):
        self.default = TraceGroup(ASCADData.random_key(), ASCADData.data_range)
        self.random = TraceGroup(ASCADData.random_key(), ASCADData.data_range, RandomTCat)
        self.masked = TraceGroup(ASCADData.random_key(), ASCADData.data_range, MaskedKeyTCat)

        # TODO these need hamming weight label as well.
        # self.desync_50 = TraceGroup(ASCADData.random_key(ASCADDataType.desync_50), ASCADData.data_range)
        # self.desync_100 = TraceGroup(ASCADData.random_key(ASCADDataType.desync_50), ASCADData.data_range)


class AscadFixedKey(TraceLoader):
    key_size = 256
    trace_len = 700
    offset = -128

    def __init__(self):
        self.default = TraceGroup(ASCADData.fixed_key(), ASCADData.data_range)
        self.random = TraceGroup(ASCADData.fixed_key(), ASCADData.data_range, RandomTCat)

