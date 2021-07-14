import numpy as np
from tqdm import tqdm

from src.tools.lists import find


def desync(sigma, raw_traces, profiling_mask, profile_traces, attack_traces):
    permutations = np.round(np.random.normal(scale=sigma, size=len(profiling_mask))).astype(int)
    trace_length = profile_traces.shape[1]

    trace_ix = find(raw_traces[0], profile_traces[0])

    profile_desync = np.ones(profile_traces.shape, dtype=profile_traces.dtype)
    attack_desync = np.ones(attack_traces.shape, dtype=attack_traces.dtype)

    progress = tqdm(total=len(profiling_mask), desc=f"Trace desynchronization, sigma={sigma}")

    ix_profile, ix_attack = 0, 0
    for ix_raw, is_profile in enumerate(profiling_mask):
        location = permutations[ix_raw] + trace_ix
        trace = raw_traces[ix_raw]

        if is_profile:
            profile_desync[ix_profile] = trace[location:trace_length + location]
            ix_profile += 1
        else:
            attack_desync[ix_attack] = trace[location:trace_length + location]
            ix_attack += 1

        progress.update(1)

    progress.close()

    return profile_desync, attack_desync
