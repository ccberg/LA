import os
from types import SimpleNamespace

DATA_ROOT = "/data/AISy"

export = {
    "tvla": os.path.join(DATA_ROOT, "aes_tvla.h5"),
    "tvla_round_5": os.path.join(DATA_ROOT, "aes_tvla_round5.h5"),
    "tvla_ttables": os.path.join(DATA_ROOT, "aes_tvla_ttables.h5")
}

aes = SimpleNamespace(**export)
