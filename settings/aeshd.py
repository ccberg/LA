import pandas as pd

AESHD_DATA = "../../data/AES_HD"


class AESHDDataType:
    traces_1 = "traces_1"
    traces_2 = "traces_2"
    traces_3 = "traces_3"
    traces_4 = "traces_4"
    traces_5 = "traces_5"
    labels = "labels"
    default = traces_1


class AESHDData:
    @staticmethod
    def random_key(data_type) -> object:
        return pd.read_csv(f"{AESHD_DATA}/{data_type}.csv")
