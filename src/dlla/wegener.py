import numpy as np
from scipy.special import comb
from scipy.stats import norm
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

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


def dlla_p_gradient_wegener(model: Model, x_attack: np.array, y_attack: np.array):
    """
    Retrieves a p-gradient from applying the trained model on the attack trace set.
    """
    total = 0
    total_correct = 0
    p_gradient = []

    min_p = 1.0
    for x, y in zip(model.predict(x_attack), y_attack):
        total += 1
        total_correct += x.argmax() == y.argmax()

        min_p = min(min_p, binomial_test(total, total_correct))
        p_gradient.append(min_p)

    return np.array(p_gradient)


def p_gradient_wegener(a, b):
    """
    Labels and z-normalizes traces, trains a MLP (as shown in the DL-LA paper).
    Returns the p-gradient for the attack traces
    """
    dlla_traces = prepare_dlla(*labelize((a, b)))
    dlla_model = make_mlp_wegener(*dlla_traces)

    return dlla_p_gradient_wegener(dlla_model, *dlla_traces[2:])
