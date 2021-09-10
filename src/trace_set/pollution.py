import math
from enum import Enum, unique


@unique
class PollutionType(Enum):
    jitter = "jitter"
    desync = "desync"
    delay = "delay"
    gauss = "gauss"


class Pollution:
    def __init__(self, pollution_type: PollutionType, parameter: float):
        self.type = pollution_type
        self.parameter = parameter

    def get_name(self):
        suffix = str(float(f"{self.parameter:.8f}")).replace('.', '-')

        return f"{self.type.name}_{suffix}"


if __name__ == '__main__':
    print(Pollution(PollutionType.delay, 100).get_name())