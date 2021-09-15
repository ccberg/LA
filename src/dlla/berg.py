from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.optimizer_v2.adam import Adam

NUM_CLASSES = 9


def build_mlp(x, y, params):
    """
    Returns a trained MLP using a tunable set of parameters
    """
    mdl = Sequential()

    assert len(params['hidden_layers']) > 1

    mdl.add(Dense(params['hidden_layers'][0], activation=params['activation'], input_shape=(x.shape[1],)))
    for nodes in params['hidden_layers'][1:]:
        mdl.add(Dense(nodes, activation=params['activation']))

    mdl.add(Dense(params['num_classes'], activation='softmax'))

    mdl.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['accuracy'])

    mdl.fit(x, y, shuffle=True, batch_size=params['batch_size'], epochs=params['epochs'], verbose=params['progress'])

    return mdl


# Params for MLP_best from Benadjila et al. (2020)
PARAMS_BEST = {
    'activation': 'relu',
    'optimizer': Adam(learning_rate=0.001),
    'losses': 'categorical_crossentropy',
    'hidden_layers': [200] * 5,
    'batch_size': 200,
    'epochs': 5,
    'progress': True,
    'num_classes': NUM_CLASSES
}


def make_mlp(x, y, num_classes=NUM_CLASSES, progress=True):
    """
    Based on MLP_best from Benadjila et al. (2020)
    """
    params = PARAMS_BEST
    params['num_classes'] = num_classes
    params['progress'] = progress

    return build_mlp(x, y, params)
