from src.aes.sbox import s_box, inv_s_box


def hw(x):
    c = 0

    while x:
        c += 1
        x &= x - 1

    return c


HAMMING_WEIGHT = [hw(x) for x in range(256)]


def sub_bytes(state):
    for i in range(len(state)):
        state[i] = s_box[state[i]]


def sub_bytes_inv(state):
    for i in range(len(state)):
        state[i] = inv_s_box[state[i]]


def add_round_key(state, round_key):
    for i in range(len(state)):
        state[i] = state[i] ^ round_key[i]


def hw_sbox_round1(state, round_key):
    """
    Hamming weight after first SBox in the first round.

    @param state: The plaintext
    @param round_key: The initial key
    """
    add_round_key(state, round_key)
    sub_bytes(state)

    return HAMMING_WEIGHT[state[0]]
