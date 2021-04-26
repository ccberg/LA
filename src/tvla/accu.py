import numpy as np
import psutil
from numpy.testing import assert_equal, assert_almost_equal
from scipy.stats import ttest_ind

from tqdm import tqdm


def make_t_test(n):
    return lambda a, b: ttest_ind(a, b, axis=0, equal_var=False)


class TvlaAccu:
    # For future enhancement, __add_batch only has the mean and variance calculated.
    num_moments = 2
    # Two types: A and B.
    num_types = 2
    # Maximum percentage of available memory that I can use.
    max_mem = .5

    def __init__(self, trace_len, make_test=make_t_test, show_progress=True):
        self.n = 0
        self.central_moments = np.zeros((self.num_types, self.num_moments + 1, trace_len), dtype=np.float128)
        self.p_gradient = [1]

        self.make_test = make_test
        self.trace_len = trace_len
        self.show_progress = show_progress
        self.last = [None, None]

    def add(self, a, b):
        free_mem = psutil.virtual_memory().available
        mem_unit = a.astype(np.float128).nbytes

        # Add batch costs about 3 #moments + index memory units.
        est_mem_use = mem_unit * (3 * self.num_moments + 1)
        est_frac = est_mem_use / (free_mem * self.max_mem)

        total_size = len(a)
        batches = np.append(np.arange(0, total_size, total_size / np.ceil(est_frac)), [total_size]).astype(int)

        if len(batches) > 1:
            for i, j in zip(batches[:-1], batches[1:]):
                self.__add_batch(a[i:j], b[i:j])
        else:
            self.__add_batch(a, b)

    def __add_batch(self, a, b):
        batch_size = len(a)
        assert batch_size == len(b) > 0

        mv_a = self.__add_type_batch(a, 0)
        mv_b = self.__add_type_batch(b, 1)

        new_n = self.n + batch_size

        min_p = self.p_gradient[-1]

        batch_ixs = range(int(self.n == 0), batch_size)
        if self.show_progress:
            batch_ixs = tqdm(batch_ixs, desc="Running t-tests")

        for i in batch_ixs:
            stat_test = self.make_test(self.n + i + 1)

            sample_ts, sample_ps = stat_test(mv_a[:, i], mv_b[:, i])

            min_p = min(min_p, np.min(sample_ps))
            self.p_gradient.append(min_p)

        self.n = new_n

    def __add_type_batch(self, x, batch_type):
        has_iv = self.n > 0
        n = self.n

        x = x.astype(np.float128)

        ix_start = n - int(has_iv) + 1
        ix_end = n + len(x) + 1

        ixs = np.repeat(np.arange(ix_start, ix_end)[:, np.newaxis], self.trace_len, axis=1)

        def prepend(arr, moment):
            if has_iv:
                cm_selected = self.central_moments[batch_type][moment]

                return np.append(arr=[cm_selected], values=arr, axis=0)
            return arr

        # Central statical moments.
        cms = np.zeros((self.num_moments + 1, *ixs.shape), dtype=np.float128)

        moment_ixs = range(self.num_moments + 1)
        if self.show_progress:
            name = "A"
            if batch_type > 0:
                name = "B"

            moment_ixs = tqdm(moment_ixs, desc=f"Calculating central moments for {name}")

        for i in moment_ixs:
            cms[i] = np.cumsum(prepend(np.power(x, i), i), axis=0)

        # This might be extended to higher order statistical moments in the future.
        x_mean = cms[1] / ixs
        x_var = cms[2] / ixs - np.power(x_mean, 2)

        self.last[batch_type] = np.array((x_mean[-1], x_var[-1]), dtype=np.float64)

        # New central moment is the central moment after inserting the last trace from this batch.
        self.central_moments[batch_type] = np.array(cms)[:, -1]

        # This might be extended to higher order statistical moments in the future.
        res = np.array((x_mean, x_var), dtype=np.float64)

        return res


if __name__ == '__main__':
    TRACE_LEN = 1400


    def gen_example(mean, trace_num, trace_len=TRACE_LEN):
        return np.random.normal(mean, 2.2, size=(trace_num, trace_len)).astype(np.uint8)


    def gen_random(mean_a, mean_b, trace_num, trace_len=TRACE_LEN):
        ex_random = np.array([gen_example(mu, trace_num) for mu in [mean_a, mean_b]]).reshape(2 * trace_num, trace_len)
        np.random.shuffle(ex_random)

        return ex_random[NUM_TRACES:], ex_random[:NUM_TRACES]


    def get_mv(x: np.array):
        return np.array((x.mean(axis=0), x.var(axis=0)))


    NUM_TRACES = 1000

    ex_a1 = gen_example(80, NUM_TRACES)
    ex_a2 = gen_example(80, NUM_TRACES)
    ex_b1 = gen_example(81, NUM_TRACES)
    ex_r1, ex_r2 = gen_random(80, 81, NUM_TRACES)

    acc = TvlaAccu(ex_a1.shape[1])
    acc.add(ex_a1, ex_b1)

    assert_equal(acc.last[0][0], ex_a1.mean(axis=0))
    assert_equal(acc.last[1][0], ex_b1.mean(axis=0))
    assert_almost_equal(acc.last[0][1], ex_a1.var(axis=0))
    assert_almost_equal(acc.last[1][1], ex_b1.var(axis=0))
