from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.optimizer_v2.adam import Adam

NUM_CLASSES = 9


def mlp_best(x, y, params):
    """
    Based on MLP_best from Benadjila et al. (2020)
    """
    mdl = Sequential()
    mdl.add(Dense(200, activation=params['activation'], input_shape=(x.shape[1],)))
    mdl.add(Dense(200, activation=params['activation']))
    mdl.add(Dense(200, activation=params['activation']))
    mdl.add(Dense(200, activation=params['activation']))
    mdl.add(Dense(200, activation=params['activation']))
    mdl.add(Dense(params['num_classes'], activation='softmax'))

    mdl.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['accuracy'])

    mdl.fit(x, y, shuffle=True, batch_size=params['batch_size'], epochs=params['epochs'],
                  verbose=params['progress'])

    return mdl


def make_mlp(x, y, num_classes=NUM_CLASSES, progress=True):
    """
    Returns a trained MLP using a tunable set of parameters
    """
    return mlp_best(x, y, {
        'activation': 'relu',
        'optimizer': Adam(learning_rate=0.001),
        'losses': 'categorical_crossentropy',
        'batch_size': 200,
        'epochs': 5,
        'progress': progress,
        'num_classes': num_classes
    })

