import math
from enum import Enum, unique


@unique
class PollutionType(Enum):
    jitter = "jitter"
    desync = "desync"


class Pollution:
    def __init__(self, pollution_type: PollutionType, parameter: float):
        self.type = pollution_type
        self.parameter = parameter

    def get_name(self):
        param = math.modf(self.parameter)
        param_name = str(int(param[1]))
        if param[0] > 0:
            param_name += f"-{str(int(param[0] * 100))}"

        return f"{self.type.name}_{param_name}"
