import numpy as np

from src.tools.plotter import plot_longform

trace_len = 50
sample_size = 350


def gen_trace(key=0, mu=128):
    return np.array([np.random.normal(mu + (key / 256 - 1), 1, trace_len) for _ in range(sample_size)], dtype=np.uint8)


def gen_mask_trace(mu=128):
    random_keys = np.random.uniform(0, 256, sample_size) / 256

    return np.array([np.random.normal(mu + (rk - 1), 1, trace_len) for rk in random_keys], dtype=np.uint8)


if __name__ == '__main__':
    tg_a1 = gen_trace(0)
    tg_a2 = gen_trace(256)
    tg_b = gen_mask_trace()

    gen_mask_trace()

    plot_longform(gen_trace(0))
    plot_longform(gen_mask_trace())
    plot_longform(gen_trace(256))
