import numpy as np
from scipy.special import comb
from scipy.stats import norm, ttest_ind
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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


def make_mlp_wegener(x, y, progress=True):
    """
    Create a Multi Layer Perceptron model as described in the paper.
    """
    trace_len = x.shape[1]

    mdl = Sequential()
    mdl.add(Dense(120, activation='relu', input_shape=(trace_len,)))
    mdl.add(Dense(90, activation='relu'))
    mdl.add(Dense(50, activation='relu'))
    mdl.add(Dense(2, activation='softmax'))

    mdl.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics='accuracy')

    mdl.fit(x, y, shuffle=True, batch_size=150, epochs=5, verbose=progress)

    return mdl


def wegener_p_gradient(model: Model, x_attack: np.array, y_attack: np.array):
    """
    Retrieves a p-gradient from applying the trained model on the attack trace set.
    """
    predictions = model.predict(x_attack).argmax(axis=1)
    labels = y_attack.argmax(axis=1)
    correct = np.array(predictions == labels)
    num_traces = len(correct)
    p_gradient = np.ones(num_traces)

    total_correct = 0
    for ix in range(num_traces):
        total = ix + 1
        total_correct += correct[ix]

        p_gradient[ix] = binomial_test(total, total_correct)

    return p_gradient


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


def wegener_p(mdl: Model, x_att: np.ndarray, y_att: np.ndarray):
    predictions = mdl.predict(x_att).argmax(axis=1)
    labels = y_att.argmax(axis=1)

    correct = np.sum(predictions == labels)
    total = len(predictions)

    return binomial_test(total, correct)


def wegener_performance(a, b):
    """
    Labels and z-normalizes traces, trains a MLP (as shown in the DL-LA paper).
    Returns the p-gradient for the attack traces
    """
    dlla_traces = prepare_dlla(*labelize((a, b)))
    dlla_model = make_mlp_wegener(*dlla_traces[:2])

    return wegener_p_gradient(dlla_model, *dlla_traces[2:])
