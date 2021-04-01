import numpy as np
from scipy.stats import t as stats_t
from scipy.stats import ttest_ind


def make_t_test(n: int, only_p=True):
    """
    Returns a t-test that takes the sample mean and variance for a list of sample points from A, and a list of sample
    points for B.
    """
    ns = np.sqrt(n)
    nmm = n - 1

    def welch_t_test(a: np.array, b: np.array):
        mean_a, var_a = np.moveaxis(a, 0, -1).astype(np.float64)
        mean_b, var_b = np.moveaxis(b, 0, -1).astype(np.float64)

        m = mean_a - mean_b
        s = np.sqrt(var_a + var_b) / ns

        t = m / s

        dof = (var_a + var_b) ** 2 / ((var_a ** 2 + var_b ** 2) / nmm)

        p = 2 * stats_t(df=dof).cdf(-np.abs(t))

        return t, p

    if only_p:
        return lambda a, b: welch_t_test(a, b)[1]

    return welch_t_test


if __name__ == '__main__':
    def gen_example(mean, trace_num, trace_len=1400):
        return np.array([[np.random.normal(mean, 2.2) for _ in range(trace_len)] for _ in range(trace_num)])


    def get_mv(x: np.array):
        return np.moveaxis(np.array((x.mean(axis=0), x.var(axis=0))), 0, -1)


    num_traces = 1000
    ex_a = gen_example(2, num_traces)
    ex_b = gen_example(2, num_traces)

    ex_a_s = get_mv(ex_a).shape

    res_sp = 1 - ttest_ind(ex_a, ex_b, axis=0, equal_var=False)[1]

    test = make_t_test(num_traces, False)

    res_custom = min(1 - test(get_mv(ex_a), get_mv(ex_b))[1])
