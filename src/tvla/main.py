import numpy as np


def device_fails(left: np.array, right: np.array, t: float):
    return np.array((left < t, right < t)).any()


def tvla(test, left, right, p=1, debug=False):
    """
    Applies some given statistical test against the given samples.

    @param test: statistical test function that takes two samples and returns test values.
    @param left: Four sample sets containing s samples representing a power trace of length t.
    @param right: Two sample sets containing s samples representing a power trace of length t.
    @param p: Decide whether a device fails using (100 * p)%-confidence interval.
    @param debug: Set to True if this function should throw Exceptions upon improperly structured input.
    @return: Whether the device fails on (A against A, A against B).
    """
    if debug:
        assert len(left) == 4
        assert len(right) == 4

    # Four different samples from distribution A.
    a, b, c, d = left
    # Two different samples from distribution B. x and y are compared to a and b, respectively.
    _, _, x, y = right

    # Test A against A.
    aa1 = test(a, c)
    aa2 = test(b, d)

    # Test A against B.
    ab1 = test(a, x)
    ab2 = test(b, y)

    t = np.percentile([aa1, aa2, ab1, ab2], p)

    if debug:
        print(t)

    # This value for t lets p% of the devices fail.
    return device_fails(aa1, aa2, t), device_fails(ab1, ab2, t)
