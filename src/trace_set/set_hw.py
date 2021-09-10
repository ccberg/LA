import os
from typing import Optional

import numpy as np

from src.data.preprocess.hw import hamming_weights
from src.tools.la import fixed_fixed, balance
from src.trace_set.abstract import AbstractTraceSet
from src.trace_set.database import Database
from src.trace_set.pollution import Pollution


class TraceSetHW(AbstractTraceSet):
    type = "hw"

    def __init__(self, database: Database, pollution: Optional[Pollution] = None,
                 limits: (Optional[int], Optional[int]) = (None, None)):
        super().__init__(database, pollution)

        self.profile_limit, self.attack_limit = limits

    def create(self, profile_traces, profile_sb, attack_traces, attack_sb):
        root_dir = os.path.dirname(self.path)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        self.open('w')

        self.add({
            "profile": {
                "traces": profile_traces,
                "state_byte": profile_sb
            },
            "attack": {
                "traces": attack_traces,
                "state_byte": attack_sb
            }
        })

        self.close()

    def __states(self, group_name: str, limit: Optional[int]):
        """
        Returns the AES states corresponding to the labels of this trace set.
        """
        f = self.open('r')

        grp = f[group_name]
        traces = np.array(grp['traces'][:limit])
        states = np.array(grp['state_byte'][:limit])

        self.close()

        return traces, states

    def __meta(self, group_name: str, limit: Optional[int]):
        f = self.open('r')

        grp = f[group_name]
        plain = np.array(grp['plaintext_byte'][:limit])
        key = np.array(grp['key_byte'][:limit])

        self.close()

        return plain, key

    def __hw(self, group_name: str, limit: Optional[int]):
        traces, states = self.__states(group_name, limit)

        return traces, hamming_weights(states)

    def __la(self, group_name: str, limit: Optional[int], balanced: bool):
        f = self.open('r')

        grp = f[group_name]

        if 'la_bit' in grp:
            traces = np.array(grp['traces'][:limit])
            la_bit = np.array(grp['la_bit'][:limit])
            self.close()
        else:
            self.close()

            all_traces, hw = self.__hw(group_name, limit)
            traces, la_bit = fixed_fixed(all_traces, hw)
            if balanced:
                traces, la_bit = balance(traces, la_bit)

        return traces, la_bit

    def profile(self):
        return self.__hw('profile', self.profile_limit)

    def attack(self):
        return self.__hw('attack', self.attack_limit)

    def profile_states(self):
        return self.__states('profile', self.profile_limit)

    def attack_states(self):
        return self.__states('attack', self.attack_limit)

    def profile_la(self, balanced=False):
        return self.__la('profile', self.profile_limit, balanced)

    def attack_la(self, balanced=False):
        return self.__la('attack', self.attack_limit, balanced)

    def profile_meta(self):
        return self.__meta('profile', self.profile_limit)

    def attack_meta(self):
        return self.__meta('attack', self.attack_limit)

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

    print(np.mean(ts.profile_la(True)[1]), np.mean(ts.profile_la()[1]))
