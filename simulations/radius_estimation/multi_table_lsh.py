"""Multi-table random-hyperplane LSH for radius estimation."""

from itertools import combinations

import numpy as np


class MultiTableLSH:
    """L independent random-hyperplane tables for radius estimation.

    Estimate(q) = mean radius over the UNION of train queries that share q's
    bucket in at least one table, expanding Hamming per table if empty.
    """

    def __init__(self, n_tables, n_bits, dim, seed):
        self.L = n_tables
        self.n_bits = n_bits
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.planes = [rng.standard_normal((dim, n_bits)) for _ in range(n_tables)]
        self._pow2 = (1 << np.arange(n_bits)).astype(np.int64)
        self.tables = None  # list of dict: code -> set(train_idx)
        self.train_radii = None

    def _codes(self, vecs, t):
        bits = (vecs @ self.planes[t]) > 0.0
        return bits.astype(np.int64) @ self._pow2

    def fit(self, train_q, train_radii):
        self.train_radii = train_radii
        self.tables = []
        for t in range(self.L):
            codes = self._codes(train_q, t)
            tab = {}
            for i, c in enumerate(codes.tolist()):
                tab.setdefault(c, []).append(i)
            self.tables.append(tab)

    def _neighbors(self, code, max_h):
        yield code
        for h in range(1, max_h + 1):
            for flips in combinations(range(self.n_bits), h):
                m = code
                for b in flips:
                    m ^= (1 << b)
                yield m

    def estimate(self, test_q, hamming, global_mean):
        n = len(test_q)
        est = np.full(n, np.nan)
        n_pooled = np.zeros(n, dtype=np.int64)
        all_codes = [self._codes(test_q, t) for t in range(self.L)]
        for i in range(n):
            pooled = set()
            for t in range(self.L):
                code = int(all_codes[t][i])
                h = hamming
                found = set()
                while True:
                    for m in self._neighbors(code, h):
                        lst = self.tables[t].get(m)
                        if lst:
                            found.update(lst)
                    if found or h >= self.n_bits:
                        break
                    h += 1
                pooled |= found
            if pooled:
                idx = np.fromiter(pooled, dtype=np.int64)
                est[i] = float(np.mean(self.train_radii[idx]))
                n_pooled[i] = len(idx)
            else:
                est[i] = global_mean
        return est, n_pooled
