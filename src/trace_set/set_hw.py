import os

import numpy as np

from src.trace_set.abstract import AbstractTraceSet
from src.trace_set.database import Database


class TraceSetHW(AbstractTraceSet):
    type = "hw"

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

    def fixed_fixed(self, test):
        pass

    def fixed_random(self, test):
        pass

    def random_random(self, test):
        pass


if __name__ == '__main__':
    TraceSetHW(Database.ascad).create(
        np.ones(1), np.ones(1), np.ones(1), np.ones(1)
    )
