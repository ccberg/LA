import numpy as np
from scipy.special import comb
from scipy.stats import norm, ttest_ind
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm

from src.dlla.preparation import prepare_dlla, labelize

NUM_CLASSES = 2


def z_test(validation_size, num_correct):
    """
    Approximation of a binomial test with success probability = 0.5.
    This simple version of the Z-test is accurate only for large sample sizes.
    """
    z_score = (num_correct - validation_size * .5) / np.sqrt(validation_size * .25)

    return norm.cdf(-z_score)


def binomial_test(validation_size, num_correct):
    """
    Binomial test with success probability = 0.5.
    Returns the p-value for hypotheses H0: validation accuracy ≤ 0.5,
        where the validation accuracy is calculated by (num_correct / validation_size).

    - validation_size refers to M from the DL-LA paper.
    - num_correct refers to s_M from the DL-LA paper.
    """
    p = 0
    for k in range(num_correct, validation_size + 1):
        # Binomial coefficient
        c = comb(validation_size, k)

        if c > 10 ** 32:
            # If the validation size is large (more than 1000 traces) p tends to go to infinite.
            # In this case, approximate the binomial test using z-test.
            return z_test(validation_size, num_correct)

        p += c

    p_value = p * .5 ** validation_size

    return p_value


def make_mlp_wegener(x, y, x_attack, y_attack):
    """
    Create a Multi Layer Perceptron model as described in the paper.
    """
    trace_len = x.shape[1]

    mdl = Sequential()
    mdl.add(Dense(120, activation='relu', input_shape=(trace_len,)))
    mdl.add(Dense(90, activation='relu'))
    mdl.add(Dense(50, activation='relu'))
    mdl.add(Dense(2, activation='softmax'))

    mdl.compile(optimizer=Adam(lr=0.001), loss='mse', metrics='accuracy')

    mdl.fit(x, y, shuffle=True, validation_data=(x_attack, y_attack), batch_size=150,
            epochs=5, verbose=True)

    return mdl


def wegener_p_gradient(model: Model, x_attack: np.array, y_attack: np.array):
    """
    Retrieves a p-gradient from applying the trained model on the attack trace set.
    """
    total = 0
    total_correct = 0
    p_gradient = []

    predictions = model.predict(x_attack).argmax(axis=1)
    labels = y_attack.argmax(axis=1)
    correct = np.array(predictions == labels)

    num_predictions = len(predictions)
    num_pts = 200000

    calc_pt = round(num_predictions / min(num_pts, num_predictions))

    p_value = 1.0
    for ix in range(num_predictions):
        total += 1
        total_correct += correct[ix]

        if ix % calc_pt == 0:
            p_value = binomial_test(total, total_correct)
        p_gradient.append(p_value)

    return np.array(p_gradient)


def wegener_t_test_p_gradient(model: Model, x_attack: np.array, y_attack: np.array):
    """
    Retrieves a p-gradient from applying the trained model on the attack trace set.
    """
    p_gradient = []

    predictions = np.average(model.predict(x_attack), axis=1, weights=range(2))
    labels = np.argmax(y_attack, axis=1).astype(bool)

    a = predictions[labels]
    b = predictions[~labels]

    num_predictions = len(predictions)

    p_value = 1.0
    for ix in tqdm(range(0, num_predictions)):
        if ix > 1:
            p_value = ttest_ind(a[:ix], b[:ix])[1]
        p_gradient.append(p_value)

    return np.array(p_gradient)


def wegener_performance(a, b):
    """
    Labels and z-normalizes traces, trains a MLP (as shown in the DL-LA paper).
    Returns the p-gradient for the attack traces
    """
    dlla_traces = prepare_dlla(*labelize((a, b)))
    dlla_model = make_mlp_wegener(*dlla_traces)

    return wegener_p_gradient(dlla_model, *dlla_traces[2:])


def balance(x, y):
    y_num = np.argmax(y, axis=1).astype(bool)

    imbalance = len(y) - 2 * np.sum(y_num)
    if imbalance < 0:
        ixs = np.where(y_num)[0]
    else:
        ixs = np.where(~y_num)[0]

    np.random.shuffle(ixs)
    drop_ixs = ixs[:abs(imbalance)]

    return np.delete(x, drop_ixs, axis=0), np.delete(y, drop_ixs, axis=0)


def class_reduction(x, y):
    """
    Takes 9-class (categorical) hamming weight labels and reduces it to 2 classes.
    """
    numerical = y.argmax(axis=1)
    filter_ixs = numerical != 4

    numerical_reduced = numerical[filter_ixs] > 4
    y2 = to_categorical(numerical_reduced).astype(np.int8)

    return balance(x[filter_ixs], y2)
