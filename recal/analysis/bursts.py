"""Burst behaviour from per-request records: no run of its own.

A request is in a burst when its query occurs at least 5 times in its 1,000-request block. The same stream is run
with the mini cache (the Q1 run) and without it (the Q2 no-mini-cache run); the burst mask splits both record sets.
"""

import numpy as np

from recal.analysis.rates import rates
from recal.robustness import burst_mask


def split(cols: dict, mask: np.ndarray) -> dict:
    sel = mask[cols["index"]]
    return {"burst": rates({k: v[sel] for k, v in cols.items()}),
            "other": rates({k: v[~sel] for k, v in cols.items()})}


def bursts_with_and_without_mini(with_mini: dict, without_mini: dict, stream_keys: np.ndarray) -> dict:
    """Hit rates and recall inside and outside bursts for the run with and the run without the mini cache."""
    mask = burst_mask(stream_keys)
    return {"with_mini": split(with_mini, mask), "without_mini": split(without_mini, mask),
            "burst_share": float(mask[with_mini["index"]].mean())}
