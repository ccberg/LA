import os

import h5py
from os import path
from src.config import DATA_ROOT

# ASCAD data root directory.
ASCAD_ROOT = path.join(DATA_ROOT, "ASCAD")
ASCAD_DATA = path.join(ASCAD_ROOT, "ATMEGA_AES_v1")

# Subdirectories, do not need to be changed if the file structure from ANSSI is maintained.
ASCAD_DATA_VAR = "/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases"
ASCAD_DATA_FIX = "/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases"


class ASCADDataType:
    raw = "ATMega8515_raw_traces"
    default = "ASCAD"


class TraceGroup:
    @staticmethod
    def random_key(data_type: str) -> h5py.File:
        pass

    @staticmethod
    def fixed_key(data_type: str) -> h5py.File:
        pass

    @staticmethod
    def raw() -> h5py.File:
        pass


class ASCADData(TraceGroup):
    @staticmethod
    def random_key(data_type=ASCADDataType.default) -> type(h5py.File):
        return h5py.File(f"{ASCAD_DATA}{ASCAD_DATA_VAR}/{data_type}.h5", 'r')

    @staticmethod
    def fixed_key(data_type=ASCADDataType.default) -> type(h5py.File):
        return h5py.File(f"{ASCAD_DATA}{ASCAD_DATA_FIX}/{data_type}.h5", 'r')

    @staticmethod
    def raw() -> type(h5py.File):
        return h5py.File(f"{ASCAD_DATA}/ASCAD_data/ASCAD_databases/atmega8515-raw-traces.h5", 'r')
