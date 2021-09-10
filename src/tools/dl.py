from tensorflow.python.keras.utils.np_utils import to_categorical


def encode(y, num_classes=9):
    """
    One-hot encode labels
    """
    return to_categorical(y, num_classes=num_classes)
