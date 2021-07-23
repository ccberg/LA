import numpy as np
from numpy import random

from src.pollution.tools import max_data


def gaussian_noise(traces: np.ndarray, noise_level: float):
    noisy = (traces + random.normal(scale=noise_level, size=traces.shape)).astype(np.int16)
    extreme = max(abs(noisy.min()), noisy.max())
    norm_factor = max_data(traces) / extreme
    if norm_factor < 1:
        return (noisy * norm_factor).astype(np.int8)

    return noisy.astype(np.int8)
