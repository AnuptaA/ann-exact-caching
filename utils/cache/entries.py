#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Cache entries: a state is a list of indices j = candidate * len(k_values) + k_index."""

import numpy as np

#-------------------------------------------------------------------------------


def entry_kmax(entries, k_values):
    """Cached candidate indices and the largest K cached for each."""
    n_kv = len(k_values)
    kmax = {}
    for j in entries:
        i, K = j // n_kv, k_values[j % n_kv]
        kmax[i] = max(kmax.get(i, 0), K)
    cand = np.array(sorted(kmax), dtype=np.int64)
    return cand, np.array([kmax[i] for i in cand], dtype=np.int64)


def perturbed(entries, frac, n_cand, n_kv, rng):
    """Replace a fraction of entries by random unselected candidates, same K."""
    entries = list(entries)
    taken = set(entries)  # never re-pick an original or an earlier replacement
    for pos in rng.choice(len(entries), int(round(frac * len(entries))), replace=False):
        t = entries[pos] % n_kv
        while True:
            j = int(rng.integers(n_cand)) * n_kv + t
            if j not in taken:
                break
        taken.add(j)
        entries[pos] = j
    return entries
