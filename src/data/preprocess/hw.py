import numpy as np
from tqdm import tqdm

from src.aes.hw_sbox import hw
from src.aes.main import AES


def full_states_hw(plaintexts: np.array, keys: np.array):
    """
    Hamming weight of all state bytes, after the SBox of each round of AES.
    """
    hws = np.zeros((len(plaintexts), 10, 16), dtype=np.uint8)

    for ix in tqdm(range(len(plaintexts))):
        aes = AES(keys[ix])
        aes.encrypt(plaintexts[ix])

        hws[ix] = [hw(i) for i in aes.state_after_round]

    return hws


def third_byte(plaintexts: np.array, keys: np.array):
    """
    Hamming weight of the third state bytes, after the SBox of each round of AES.
    """
    return full_states_hw(plaintexts, keys)[:, :, 2]


BYTE_LEN = 256


def full_states(plaintexts: np.array, keys: np.array):
    """
    All state bytes, after the SBox of each round of AES-128.
    """
    states = np.zeros((len(plaintexts), 10, 16), dtype=np.uint8)

    for ix in tqdm(range(len(plaintexts))):
        aes = AES(keys[ix])
        aes.encrypt(plaintexts[ix])

        states[ix] = aes.state_after_round

    return states


def hamming_weights(state_list: np.ndarray):
    """
    Constructs a hamming weight lookup table, which it uses to transform an array of state bytes
        into their hamming weights.
    """
    hw_lookup = np.zeros(BYTE_LEN, dtype=np.uint8)
    for i in range(BYTE_LEN):
        hw_lookup[i] = hw(i)

    return hw_lookup[state_list]
