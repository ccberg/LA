import pandas as pd

ASCAD_DATA = "../../data/ASCAD/ATMEGA_AES_v1"
ASCAD_DATA_VAR = "/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases"
ASCAD_DATA_FIX = "/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases"


class ASCADDataType:
    raw = "ATMega8515_raw_traces"
    default = "ASCAD"
    desync_50 = "ASCAD_desync50"
    desync_100 = "ASCAD_desync100"


class ASCADData:
    @staticmethod
    def random_key(data_type) -> object:
        return pd.read_hdf(f"{ASCAD_DATA}{ASCAD_DATA_VAR}/{data_type}.h5")

    @staticmethod
    def fixed_key(data_type) -> object:
        return pd.read_hdf(f"{ASCAD_DATA}{ASCAD_DATA_FIX}/{data_type}.h5")
