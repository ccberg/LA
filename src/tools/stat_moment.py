import numpy as np

from src.data.dummy.trace_generation import gen_trace


def get_mvs(samples):
    return np.array([np.moveaxis(np.array([s.mean(axis=0), s.var(axis=0)]), 0, 1) for s in samples.astype(np.float128)])


if __name__ == '__main__':
    sps = np.array([gen_trace(5), gen_trace(5)])
    print(get_mvs(sps).shape)
