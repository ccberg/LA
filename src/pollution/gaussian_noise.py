import numpy as np
from numpy import random


def gaussian_noise(traces: np.ndarray, noise_level: float):
    """
    Based on the implementation of L. Wu & S. Picek (2020): "Remove Some Noise: On Pre-processing of Side-channel
        Measurements with Autoencoders."
    """
    return traces + random.normal(scale=noise_level, size=traces.shape)
