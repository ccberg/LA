import math

import numpy as np
from scipy.integrate import quad

DEN_CACHE, EXP_CACHE, FUN_CACHE = {}, {}, {}


def make_f_chi(v: int):
    def f_chi(x: float) -> float:
        if x <= 0:
            return 0.0

        num = x ** EXP_CACHE[v] * math.e ** (x * -.5)
        return num * DEN_CACHE[v]

    return f_chi


for i in range(1, 64 + 1):
    DEN_CACHE[i] = 1 / (2 ** (i / 2) * math.gamma(i / 2))
    EXP_CACHE[i] = ((i / 2) - 1)
    FUN_CACHE[i] = make_f_chi(i)

FUN_CACHE[0] = lambda _: 0.0


def p_chi(sum_chi, dof):
    # noinspection PyTypeChecker
    return quad(FUN_CACHE[dof], sum_chi, np.inf)[0]


NUM_CATEGORIES = 2
NUM_CAT_DOF = NUM_CATEGORIES - 1
RANGE_CAT = range(NUM_CATEGORIES)


def chi_squared(observed, expected):
    """
    Calculates the p value for rejecting H0.
    Small p values give evidence to reject the null hypothesis and conclude that for the
    scenarios presented in ctable the occurrences of the observations are not independent.
    """
    categories = np.array([observed, expected])

    # Bins where both categories are > 0
    nz = categories.any(axis=0)

    sum_oe = (np.array(observed) + np.array(expected))[nz]
    ef = [sum_oe * .5] * 2
    chi_acc = ((categories[:, nz] - ef) ** 2 / ef).sum()

    # Degrees of freedom. Length of sum_oe is faster than summing nz.
    dof = (len(sum_oe) - 1) * NUM_CAT_DOF

    # return chi_acc, dof
    return 1 - p_chi(chi_acc, dof)


def chi2(lefts, rights, p):
    return np.array([chi_squared(l, r) for l, r in zip(lefts, rights)]) > p