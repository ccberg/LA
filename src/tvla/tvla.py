
import numpy as np
from matplotlib.pyplot import show

from tqdm import tqdm
import seaborn as sns
from src.tools.plotter import plot_p_gradient, PALETTE_GRADIENT
from src.tvla.t import make_t_test


def central_sum(x, order):
    mean_free = x - x.mean(axis=0)

    return np.sum(mean_free ** order, axis=0)


def central_moment(x, order):
    if order <= 1:
        return np.zeros(x.shape[1])
    else:
        return central_sum(x, order) / len(x)


class Group:
    def __init__(self, traces: np.array, max_order=3, progress=False):
        self.mean = traces.mean(axis=0)
        mean_free = traces - self.mean

        max_computed_order = 2 * (max_order + 1)

        self.num_traces, self.trace_len = len(traces), 1
        if len(traces.shape) == 2:
            self.num_traces, self.trace_len = traces.shape

        shape = (max_computed_order, self.trace_len)

        self.cm, self.cm2 = np.zeros(shape), np.zeros(shape)

        computed_orders = np.unique([*range(max_order + 1), *(2 * np.arange(max_order + 1))])

        if progress:
            computed_orders = tqdm(computed_orders, "Computing Central Moments")

        for order in computed_orders:
            cm = np.sum(mean_free ** order, axis=0) / self.num_traces

            self.cm[order] = cm
            self.cm2[order] = cm ** 2

    def __sm(self, order):
        if order == 1:
            return self.mean
        if order == 2:
            return self.cm[2]
        if order > 2:
            return self.cm[order] / (self.cm[2] ** (order / 2))

    def __s2(self, order):
        if order == 1:
            return self.cm[2]
        if order == 2:
            return self.cm[4] - (self.cm2[2])
        if order > 2:
            return (self.cm[order * 2] - self.cm2[order]) / (self.cm[2] ** order)

    def t_estimates(self, order):
        return self.__sm(order), self.__s2(order)

    def t_test(self, other: 'Group', order: int):
        if self.num_traces <= 2:
            return np.zeros(self.trace_len), np.ones(self.trace_len)

        t = make_t_test(self.num_traces, other.num_traces)
        return t(*self.t_estimates(order), *other.t_estimates(order))


def peek(a: list, default=None):
    if a:
        return a[-1]
    return default


class Tvla:
    def __init__(self, trace_len, max_order=3, gradient_pts=200):
        self.trace_len = trace_len
        self.max_order = max_order
        self.min_p_gradient = None
        self.min_p = None
        self.gradient_pts = gradient_pts

    def __set_min_p(self, a, b):
        group_a, group_b = Group(a, self.max_order, True), Group(b, self.max_order, True)
        self.min_p = np.ones((self.max_order + 1, self.trace_len))

        min_p_ixs = np.ones(self.max_order + 1, dtype=int)
        for order in range(1, self.max_order + 1):
            p_values = group_a.t_test(group_b, order)[1]
            self.min_p[order] = p_values
            min_p_ixs[order] = np.argmin(np.nan_to_num(p_values, nan=1.0))

        return min_p_ixs

    def __set_min_p_gradient(self, a, b, min_p_ixs):
        min_num_traces = min(len(a), len(b))
        a, b = a.copy(), b.copy()

        self.min_p_gradient = np.ones((self.max_order + 1, min_num_traces * 2))

        for order in tqdm(range(1, self.max_order + 1), "Computing min-p gradients"):
            selected_a = a[:, min_p_ixs[order]]
            selected_b = b[:, min_p_ixs[order]]

            for trace_ix in range(1, min_num_traces):
                sp_group_a = Group(selected_a[:trace_ix], order)
                sp_group_b = Group(selected_b[:trace_ix], order)

                p_value = sp_group_a.t_test(sp_group_b, order)[1][0]

                # 2 Traces are added every iteration.
                p_value_ix = trace_ix * 2
                self.min_p_gradient[order, p_value_ix] = p_value
                self.min_p_gradient[order, p_value_ix + 1] = p_value

    def add(self, a, b):
        # Copy traces, they will be shuffled.
        a, b = a.copy(), b.copy()

        min_p_ixs = self.__set_min_p(a, b)
        np.random.shuffle(a), np.random.shuffle(b)
        self.__set_min_p_gradient(a, b, min_p_ixs)

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

    def plot_gradients(self):
        lines = dict([(f"$\\mu_{{{d}}}$", self.p_gradient(d)) for d in range(1, self.max_order + 1)])
        plot_p_gradient(lines, "TVLA $p$-gradients for statistical\nmoment orders $\\mu_d$\n", palette=PALETTE_GRADIENT)

    def plot_min_p(self, order):
        g = sns.lineplot(data={f"$\\mu_{{{order}}}$": self.min_p[order]})
        sns.lineplot(data={"Threshold": np.ones(len(self.min_p[order])) * 10 ** -5},
                     palette=["red"], dashes=[(2, 2)])
        g.set(yscale="log", ylabel="$p$-value for dist. $A \\neq$ dist. $B$", xlabel="Sample point",
              title=f"Min-$p$ values for $\\mu_{{{order}}}$", ylim=(None, 1))

        g.invert_yaxis()
        show(block=False)
