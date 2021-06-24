import math
import os
from enum import Enum

import h5py
import numpy as np

from src.tools.constants import DATA_DIR


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


class Database(Enum):
    ascad = "ascad"
    aisy = "aisy"


class TraceSet:
    class Column(Enum):
        traces = "traces"
        hamming_weight = "hamming_weight"

    def __init__(self, database: Database, pollution: Pollution = None):
        dir_root = os.path.join(DATA_DIR, database.name, "generic")
        self.h5 = None

        if pollution is None:
            self.name = "default"
        else:
            self.name = pollution.get_name()

        self.path = os.path.join(dir_root, f"{self.name}.h5")

    def open(self, permission="r"):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, permission)
        else:
            raise IOError("Trace set file already opened")

        return self.h5

    def close(self):
        if self.h5 is not None:
            self.h5.close()
        else:
            raise IOError("Trace set file not opened")

        self.h5 = None

    def add(self, data, name: str = None, group=None):
        t = type(data)
        if t is dict:
            for group_name, group_data in data.items():
                if group is None:
                    group = self.h5.require_group(group_name)

                self.add(group_data, group_name, group)
        elif t is np.ndarray:
            if group is not None:
                if name in group:
                    del group[name]

                group[name] = data
            else:
                raise TypeError(f"No root group for numpy array to fit in.", )
        else:
            raise TypeError(f"Unable to put {t} in h5 file.", )

    def fixed_fixed(self):
        raise NotImplementedError()

    def fixed_random(self):
        raise NotImplementedError()

    def random_random(self):
        raise NotImplementedError()
