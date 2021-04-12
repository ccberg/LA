import numpy as np
from tqdm import tqdm

from src.aes.main import AES


def full_states(plaintexts: np.array, keys: np.array):
    """
    Hamming weight of all state bytes, after the SBox of each round of AES.
    """
    hws = np.zeros((len(plaintexts), 10, 16), dtype=np.uint8)

    for ix in tqdm(range(len(plaintexts))):
        aes = AES(keys[ix])
        aes.encrypt(plaintexts[ix])

        hws[ix] = aes.hw_after_round

    return hws


def third_byte(plaintexts: np.array, keys: np.array):
    """
    Hamming weight of the third state bytes, after the SBox of each round of AES.
    """
    return full_states(plaintexts, keys)[:, :, 2]

