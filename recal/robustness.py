"""Burst marking for the robustness analyses."""

import numpy as np

def burst_mask(keys: np.ndarray, block: int = 1000, min_count: int = 5) -> np.ndarray:
    """A request is in a burst when its query occurs at least min_count times in its block of requests."""
    keys = np.asarray(keys)
    out = np.zeros(len(keys), dtype=bool)
    for s in range(0, len(keys), block):
        _, inverse, counts = np.unique(keys[s:s + block], return_inverse=True, return_counts=True)
        out[s:s + block] = counts[inverse] >= min_count
    return out


def split_by_mask(cols: dict, mask: np.ndarray) -> dict:
    """Per-request records restricted to the requests with mask true and false (aligned with cols['index'])."""
    from recal.analysis.rates import rates
    index = cols["index"]
    sel = mask[index]
    return {"burst": rates({k: v[sel] for k, v in cols.items()}),
            "other": rates({k: v[~sel] for k, v in cols.items()})}
