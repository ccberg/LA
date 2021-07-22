import os

import numpy as np

from src.trace_set.abstract import AbstractTraceSet
from src.trace_set.database import Database
from src.trace_set.pollution import Pollution


class TraceSetHW(AbstractTraceSet):
    type = "hw"

    def __init__(self, database: Database, pollution: Pollution = None, limits: (int, int) = (None, None)):
        super().__init__(database, pollution)

        self.profile_limit, self.attack_limit = limits

    def create(self, profile_traces, profile_hw, attack_traces, attack_hw):
        root_dir = os.path.dirname(self.path)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        self.open('w')

        self.add({
            "profile": {
                "traces": profile_traces,
                "hw": profile_hw
            },
            "attack": {
                "traces": attack_traces,
                "hw": attack_hw
            }
        })

        self.close()

    def __fetch(self, group_name: str, limit: int):
        f = self.open('r')

        grp = f[group_name]
        res = np.array(grp['traces'][:limit]), np.array(grp['hw'][:limit])

        self.close()

        return res

    def profile(self):
        return self.__fetch('profile', self.profile_limit)

    def attack(self):
        return self.__fetch('attack', self.attack_limit)

    def all(self):
        px, py = self.profile()
        ax, ay = self.attack()

        # Create accumulators.
        num_traces = len(px) + len(ax)
        len_traces = px.shape[1]
        traces, labels = np.zeros((num_traces, len_traces), dtype=px.dtype), np.zeros(num_traces, dtype=py.dtype)

        # Concatenate profiling and attack traces.
        traces[:len(px)], labels[:len(py)] = px, py
        traces[len(px):], labels[len(py):] = ax, ay

        return traces, labels


if __name__ == '__main__':
    ts = TraceSetHW(Database.aisy)

    print(ts.all())
