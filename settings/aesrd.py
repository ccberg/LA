import pandas as pd
import numpy as np
import h5py

AESRD_DATA = "../../data/AES_RD"


class AESRDDataType:
    default = "ctraces_fm16x4_2"


class AESRDData:
    @staticmethod
    def fixed_key(data_type) -> object:
        f = h5py.File('somefile.mat', 'r')
        data = f.get('data/variable1')
        data = np.array(data)  # For converting to a NumPy array

        return pd.read_csv(f"{AESRD_DATA}/{data_type}.csv")
