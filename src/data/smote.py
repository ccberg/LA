import numpy as np


def smote_slice(trace_slice, target_size):
    n = len(trace_slice)

    smote_num = target_size - n
    smote_dist = np.random.uniform(size=(smote_num, trace_slice.shape[1]))

    s1 = np.random.randint(n, size=smote_num)
    s2 = (s1 + np.random.randint(1, n, size=smote_num)) % n

    d = (np.array(trace_slice[s1], dtype=int) - np.array(trace_slice[s2], dtype=int)) * smote_dist
    app = np.array(np.round(trace_slice[s2] + d), dtype=np.uint8)

    return app


def smote(trace_categories, target_size, dtype=np.uint8):
    acc = []

    for c_ix, category in enumerate(trace_categories):
        res = np.zeros((len(category), target_size, category[0].shape[1]), dtype=dtype)

        for s_ix, trace_slice in enumerate(category):
            app = smote_slice(trace_slice, target_size)

            if len(app) > 0:
                res[s_ix] = np.concatenate((trace_slice, app))
            else:
                res[s_ix] = trace_slice

        acc.append(res)

    return acc
