import numpy as np

from tqdm import tqdm

from src.tvla.t import make_t_test


# TODO make group
def central_sum(x, order):
    mean_free = x - x.mean(axis=0)

    return np.sum(mean_free ** order, axis=0)


def central_moment(x, order):
    if order <= 1:
        return np.zeros(x.shape[1])
    else:
        return central_sum(x, order) / len(x)


def sm(x, order):
    if order == 1:
        return np.mean(x, axis=0)
    if order == 2:
        return central_moment(x, 2)
    if order > 2:
        return central_moment(x, order) / (central_moment(x, 2) ** (order / 2))


def s2(x, order):
    if order == 1:
        return central_moment(x, 2)
    if order == 2:
        return central_moment(x, 4) - (central_moment(x, 2) ** 2)
    if order > 2:
        return (central_moment(x, order * 2) - (central_moment(x, order) ** 2)) / (central_moment(x, 2) ** order)


def order_test(n: int, order: int):
    t = make_t_test(n)

    def t_test(a: np.array, b: np.array):
        ma, va = sm(a, order), s2(a, order)
        mb, vb = sm(b, order), s2(b, order)

        return t(ma, va, mb, vb)

    return t_test


class Tvla:
    def __init__(self, trace_len, max_order=3):
        self.trace_len = trace_len
        self.max_order = max_order
        self.min_p_gradient = dict([(i, []) for i in range(max_order + 1)])
        self.min_p = np.ones((max_order + 1, trace_len))

    def add(self, a, b):
        for ix in tqdm(range(min(len(a), len(b)))):
            self.min_p_gradient[0].append(1.0)
            for d in range(1, self.max_order + 1):
                t = order_test(ix, d)

                if ix > 2:
                    res = t(a[:ix], b[:ix])[1]
                    self.min_p[d] = np.minimum(self.min_p[d], res)

                self.min_p_gradient[d].append(min(self.min_p[d]))

    def __assert_order(self, order):
        if order > self.max_order:
            raise ValueError(f"Provided order ({order}) is larger than max order ({self.max_order}).")

    def p_gradient(self, order):
        self.__assert_order(order)
        return self.min_p_gradient[order]

    def min_p(self, order):
        self.__assert_order(order)
        return self.min_p[order]

    def min_p_order(self):
        return [self.min_p_gradient[i][-1] for i in self.min_p_gradient]
