import numpy as np


def smote(samples: np.array, preferred_size: int = -1):
    if preferred_size < 0:
        preferred_size = max([len(s) for s in samples])

    # Create result array in correct shape (number of samples = preferred size).
    res_shape = (len(samples), preferred_size, len(samples[0][0]))
    res = np.zeros(res_shape)

    for ix, sample_slice in enumerate(samples):
        num_smote = preferred_size - len(sample_slice)
        assert num_smote >= 0

        ix_left = np.random.uniform(0, len(sample_slice), num_smote).astype(int)
        ix_right = (ix_left + np.random.uniform(1, len(sample_slice), num_smote).astype(int)) % len(sample_slice)

        dist = sample_slice[ix_left] - sample_slice[ix_right]
        smote_delta = dist * np.random.uniform(size=dist.shape)
        smote_res = np.round(sample_slice[ix_right] + smote_delta).astype(int)

        res[ix][:len(sample_slice)] = sample_slice
        res[ix][len(sample_slice):] = smote_res

    return res
