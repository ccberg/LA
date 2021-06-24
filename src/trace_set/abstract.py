import os

import h5py
import numpy as np

from src.tools.constants import DATA_DIR
from src.trace_set.database import Database
from src.trace_set.pollution import Pollution


class AbstractTraceSet:
    type = None

    def __init__(self, database: Database, pollution: Pollution = None):
        dir_root = os.path.join(DATA_DIR, database.name, self.get_type())
        self.h5 = None

        if pollution is None:
            self.name = "default"
        else:
            self.name = pollution.get_name()

        self.path = os.path.join(dir_root, f"{self.name}.h5")

    def get_type(self):
        if self.type is None:
            raise NotImplementedError()
        else:
            return self.type

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

    def fixed_random(self):
        raise NotImplementedError()

    def random_random(self):
        raise NotImplementedError()
