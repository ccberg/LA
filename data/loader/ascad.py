import h5py

ASCAD_DATA = "/data/ASCAD/ATMEGA_AES_v1"
ASCAD_DATA_VAR = "/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases"
ASCAD_DATA_FIX = "/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases"

# Range of the interesting part of the trace, selected by ASCAD.
# Not sure if this range also holds for the raw_traces datasets.
ASCAD_SELECTED_DATA_RANGE = range(-128, 127)


class ASCADDataType:
    raw = "ATMega8515_raw_traces"
    default = "ASCAD"
    desync_50 = "ASCAD_desync50"
    desync_100 = "ASCAD_desync100"


class TraceGroup:
    data_range = []

    @staticmethod
    def random_key(data_type: str) -> h5py.File:
        pass

    @staticmethod
    def fixed_key(data_type: str) -> h5py.File:
        pass


class ASCADData(TraceGroup):
    data_range = ASCAD_SELECTED_DATA_RANGE

    @staticmethod
    def random_key(data_type=ASCADDataType.default) -> object:
        return h5py.File(f"{ASCAD_DATA}{ASCAD_DATA_VAR}/{data_type}.h5", 'r')

    @staticmethod
    def fixed_key(data_type=ASCADDataType.default) -> object:
        return h5py.File(f"{ASCAD_DATA}{ASCAD_DATA_FIX}/{data_type}.h5", 'r')
