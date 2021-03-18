import numpy as np


def smote_slices(samples: np.array, preferred_size: int = -1):
    if preferred_size < 0:
        preferred_size = max([len(s) for s in samples])

    # Create result array in correct shape (number of samples = preferred size).
    res_shape = (len(samples), preferred_size, len(samples[0][0]))
    res = np.zeros(res_shape)

    for ix, sample_slice in enumerate(samples):
        smote_res = do_smote(samples, preferred_size)

        res[ix][:len(sample_slice)] = sample_slice
        res[ix][len(sample_slice):] = smote_res

    return res


def do_smote(samples, preferred_size):
    """
    Applies SMOTE to a set of samples.
    """
    num_smote = preferred_size - len(samples)
    assert num_smote >= 0

    # First set of indexes.
    ix_left = np.random.uniform(0, len(samples), num_smote).astype(int)
    # Second set of indexes, indexes should not overlap with first set of indexes.
    ix_right = (ix_left + np.random.uniform(1, len(samples), num_smote).astype(int)) % len(samples)

    # Distance between first and second set of indexes.
    dist = samples[ix_left] - samples[ix_right]
    smote_delta = dist * np.random.uniform(size=dist.shape)

    return np.round(samples[ix_right] + smote_delta).astype(int)


def smote(samples: np.array, preferred_size: int):
    res_shape = (preferred_size, *samples[0].shape)
    res = np.zeros(res_shape)

    smote_res = do_smote(samples, preferred_size)

    res[:len(samples)] = samples
    res[len(samples):] = smote_res

    return res


if __name__ == '__main__':
    smoted = smote(np.zeros((10, 1400)), 100)
    print(smoted.shape)
