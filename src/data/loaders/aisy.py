import os
from src.config import DATA_ROOT
from types import SimpleNamespace

AISY_ROOT = os.path.join(DATA_ROOT, "AISy")

export = {
    "tvla": os.path.join(AISY_ROOT, "aes_tvla.h5"),
    "tvla_round_5": os.path.join(AISY_ROOT, "aes_tvla_round5.h5"),
    "tvla_ttables": os.path.join(AISY_ROOT, "aes_tvla_ttables.h5")
}

aes = SimpleNamespace(**export)
