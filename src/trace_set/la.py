import os

from src.trace_set.abstract import AbstractTraceSet


class LATraceSet(AbstractTraceSet):
    type = "la"

    def create(self, profile_traces, profile_la_bit, attack_traces, attack_la_bit):
        root_dir = os.path.dirname(self.path)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        self.open('w')

        self.add({
            "profile": {
                "traces": profile_traces,
                "la_bit": profile_la_bit
            },
            "attack": {
                "traces": attack_traces,
                "la_bit": attack_la_bit
            }
        })

        self.close()

    def fixed_random(self):
        pass

    def random_random(self):
        pass
