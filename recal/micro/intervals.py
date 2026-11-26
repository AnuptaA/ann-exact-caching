"""Exact score intervals for window queries kept at low precision (Part 2).

A window query keeps a quantized copy q_hat and its error e = |q - q_hat| (stored in 2 bytes, rounded up), so
|d(q, x) - d(q_hat, x)| <= e for every point x. The v11 features are computed with exact searches:
  nr       distance to the K-th nearest cached vector (the candidate answer is the exact in-pool top-K)
  fit      best-fit entry among the c nearest centers: min (d(q, c_j) + nr) / r_j, offset and size of that entry
  rate     log((count + 1) / K), count = shell vectors of those entries (4-bit codes decoded as the core does)
                strictly closer than nr, each shell vector counted once
Intervals follow from the Lipschitz bound; the entry set itself depends on q, so entries that may or must be among
the c nearest are bracketed too. A linear score takes each feature's lower or upper end by the sign of its
coefficient. certify_worst_case certifies lambda on the worst served set consistent with the intervals.
"""

from statistics import NormalDist

import numpy as np

from recal.micro import e4, intruder_check, topk, world as world_mod
from recal.micro.quantize import quantize

BLOCK_ELEMENTS = 20_000_000
TOLERANCE = 1e-9


def quantize_with_error(X: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """(q_hat, e): the vectors after a round trip through `bits` bits (16: float16) and the stored error bound."""
    X = np.asarray(X, dtype=np.float32)
    Xq = X.astype(np.float16).astype(np.float32) if bits == 16 else quantize(X, bits)
    err = np.linalg.norm(X.astype(np.float64) - Xq.astype(np.float64), axis=1)
    stored = err.astype(np.float16)
    stored = np.where(stored.astype(np.float64) < err, np.nextafter(stored, np.float16(np.inf)), stored)
    return Xq, stored.astype(np.float64)


def _dist(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    Q, X = Q.astype(np.float64), X.astype(np.float64)
    return np.sqrt(np.maximum((Q ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2.0 * Q @ X.T, 0.0))


class Features:
    """v11 features of queries against a built cache, exactly and as intervals."""

    def __init__(self, world, c: int = 32):
        self.world = world
        self.k, self.c = world.k, c
        self.table = e4.EntryTable(world, exclude_pool=True)
        t = self.table
        self.centers, self.radius, self.pool = t.centers, t.radius, t.pool
        self.shell_vec = intruder_check.decode_codes(world.core.shell_codes(), *world.core.quantizer())
        slot_ids = intruder_check.shell_slot_ids(world, t)
        entries = world.entries()
        self.slots = [np.asarray(e["shell_slots"], dtype=np.int64) for e in entries]
        self.pool_ids = t.pool_ids
        self.slot_ids = slot_ids

    @property
    def workers(self) -> int:
        return getattr(getattr(self, "world", None), "workers", 1)

    def _shell_union(self, entries: np.ndarray) -> np.ndarray:
        if not len(entries):
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate([self.slots[e] for e in entries]))

    def _blocks(self, n: int):
        width = max(len(self.pool), len(self.centers), len(self.shell_vec), 1)
        step = max(1, BLOCK_ELEMENTS // width)
        for lo in range(0, n, step):
            yield lo, min(n, lo + step)

    def candidate_recall(self, Q: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """has and the recall (by id) of the exact in-pool top-K answer, ties to the lower id like the core."""
        n = len(Q)
        has = np.full(n, len(self.pool) >= self.k)
        if not has.any():
            return has, np.zeros(n)
        return has, np.concatenate(world_mod.fork_map(lambda lo, hi: self._recall_range(Q[lo:hi], truth[lo:hi]),
                                                      n, self.workers))

    def _recall_range(self, Q: np.ndarray, truth: np.ndarray) -> np.ndarray:
        recall = np.zeros(len(Q))
        for lo, hi in self._blocks(len(Q)):
            Dp = _dist(Q[lo:hi], self.pool)
            for i in range(hi - lo):
                near, _ = topk.exact_topk(self.pool, Q[lo + i], Dp[i], self.k)
                recall[lo + i] = np.isin(truth[lo + i][:self.k], self.pool_ids[near]).mean()
        return recall

    def exact(self, Q: np.ndarray, nr: np.ndarray | None = None, sets: list | None = None) -> np.ndarray:
        """Features (offset, size, rate) of full-precision queries, one row each (NaN without a candidate).

        nr overrides the distance to the K-th cached vector (a fast-path answer's), sets the entries read."""
        n = len(Q)
        if len(self.pool) < self.k:
            return np.full((n, 3), np.nan)
        return np.concatenate(world_mod.fork_map(
            lambda lo, hi: self._exact_range(Q[lo:hi], None if nr is None else nr[lo:hi],
                                             None if sets is None else sets[lo:hi]), n, self.workers))

    def _exact_range(self, Q: np.ndarray, nr, sets) -> np.ndarray:
        n, K = len(Q), self.k
        out = np.full((n, 3), np.nan)
        c = min(self.c, len(self.centers))
        for lo, hi in self._blocks(n):
            Dc = _dist(Q[lo:hi], self.centers)
            Dp = _dist(Q[lo:hi], self.pool) if nr is None else None
            Ds = _dist(Q[lo:hi], self.shell_vec) if len(self.shell_vec) else np.zeros((hi - lo, 0))
            for i in range(hi - lo):
                g = lo + i
                if nr is None:
                    r = max(float(topk.exact_topk(self.pool, Q[g], Dp[i], K)[1][-1]), 1e-12)
                else:
                    r = max(float(nr[g]), 1e-12)
                S = np.argpartition(Dc[i], c - 1)[:c] if sets is None else np.asarray(sets[g], dtype=np.int64)
                fit = (Dc[i][S] + r) / self.radius[S]
                b = int(S[np.argmin(fit)])
                slots = self._shell_union(S)
                count = int((Ds[i][slots] < r).sum()) if len(slots) else 0
                out[g] = (Dc[i][b] / self.radius[b], r / self.radius[b], np.log((count + 1.0) / K))
        return out

    def bounds(self, Qhat: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper ends of (offset, size, rate) of the true queries, from their copies and errors."""
        n = len(Qhat)
        if len(self.pool) < self.k:
            return np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        parts = world_mod.fork_map(lambda lo, hi: self._bounds_range(Qhat[lo:hi], e[lo:hi]), n, self.workers)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def _bounds_range(self, Qhat: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, K = len(Qhat), self.k
        lower, upper = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        c = min(self.c, len(self.centers))
        for lo, hi in self._blocks(n):
            Dc, Dp = _dist(Qhat[lo:hi], self.centers), _dist(Qhat[lo:hi], self.pool)
            Ds = _dist(Qhat[lo:hi], self.shell_vec) if len(self.shell_vec) else np.zeros((hi - lo, 0))
            for i in range(hi - lo):
                err = float(e[lo + i])
                nr = max(float(topk.exact_topk(self.pool, Qhat[lo + i], Dp[i], K)[1][-1]), 1e-12)
                nr_lo, nr_hi = max(nr - err, 1e-12), nr + err
                dc = float(np.partition(Dc[i], c - 1)[c - 1])
                if c >= len(self.centers):
                    possible = certain = np.arange(len(self.centers))
                else:
                    possible = np.flatnonzero(Dc[i] <= dc + 2.0 * err)
                    certain = np.flatnonzero(Dc[i] < dc - 2.0 * err)
                fit = (Dc[i] + nr) / self.radius
                fit_low = fit[possible] - 2.0 * err / self.radius[possible]
                best_up = float((fit[certain] + 2.0 * err / self.radius[certain]).min()) if len(certain) else np.inf
                keep = possible[fit_low <= best_up]
                off_lo = np.maximum(Dc[i][keep] - err, 0.0) / self.radius[keep]
                off_hi = (Dc[i][keep] + err) / self.radius[keep]
                lower[lo + i, 0], upper[lo + i, 0] = off_lo.min(), off_hi.max()
                lower[lo + i, 1] = (nr_lo / self.radius[keep]).min()
                upper[lo + i, 1] = (nr_hi / self.radius[keep]).max()
                slots_p, slots_c = self._shell_union(possible), self._shell_union(certain)
                count_lo = int((Ds[i][slots_c] < nr - 2.0 * err).sum()) if len(slots_c) else 0
                count_hi = int((Ds[i][slots_p] < nr + 2.0 * err).sum()) if len(slots_p) else 0
                lower[lo + i, 2], upper[lo + i, 2] = np.log((count_lo + 1.0) / K), np.log((count_hi + 1.0) / K)
        return lower, upper


def score_interval(model: dict, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear score range: each feature at the end its coefficient's sign selects."""
    coef = np.asarray(model["coef"], dtype=np.float64)
    zl, zu = (lower - model["mean"]) / model["scale"], (upper - model["mean"]) / model["scale"]
    low = np.where(coef >= 0, zl, zu) @ coef + model["intercept"]
    high = np.where(coef >= 0, zu, zl) @ coef + model["intercept"]
    return low, high


def worst_case_set(y: np.ndarray, w: np.ndarray, certain: np.ndarray, ambiguous: np.ndarray) -> np.ndarray:
    """Served set that minimizes the weighted mean recall: the certain items plus the ambiguous items, lowest
    recall first, while each one is below the running mean. Returns a boolean mask over all items."""
    served = certain.copy()
    items = np.flatnonzero(ambiguous)
    if not len(items):
        return served
    items = items[np.argsort(y[items], kind="stable")]
    total0, weighted0 = float(w[served].sum()), float((w[served] * y[served]).sum())
    cw, cwy = np.cumsum(w[items]), np.cumsum(w[items] * y[items])
    before_w = total0 + np.r_[0.0, cw[:-1]]
    before_wy = weighted0 + np.r_[0.0, cwy[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        below = (before_w <= 0) | (y[items] < before_wy / before_w)
    take = len(items) if below.all() else int(np.argmin(below))
    served[items[:take]] = True
    return served


def lower_bound(y: np.ndarray, w: np.ndarray, z: float) -> dict:
    """v10's lower confidence bound of the weighted mean recall of a served set."""
    W, S1, S2, SW2 = w.sum(), (w * y).sum(), (w * y * y).sum(), (w * w).sum()
    if W <= 0 or SW2 <= 0:
        return dict(ess=0.0, mean=float("nan"), lcb=float("nan"), slack=float("nan"))
    mean, ess = S1 / W, W * W / SW2
    sd = np.sqrt(max(S2 - W * mean ** 2, 0.0) / max(W - SW2 / W, 1e-12))
    slack = z * sd / np.sqrt(ess)
    return dict(ess=float(ess), mean=float(mean), lcb=float(mean - slack), slack=float(slack))


def certify_worst_case(edges: np.ndarray, low: np.ndarray, high: np.ndarray, y: np.ndarray, target: float,
                       delta: float, min_ess: float, weights: np.ndarray | None = None) -> dict:
    """v10's scan over the threshold grid (strictest first, skip low ESS, stop at the first failure) with the
    worst-case served set at each threshold. Returns lam (None if uncertified) and the set's statistics."""
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    z = NormalDist().inv_cdf(1.0 - delta)
    best = dict(lam=None, ess=0.0, mean=float("nan"), lcb=float("nan"), slack=float("nan"), mean_possible=float("nan"),
                n_ambiguous=0)
    for lam in np.asarray(edges)[::-1]:
        certain = low >= lam
        ambiguous = (high >= lam) & ~certain
        served = worst_case_set(y, w, certain, ambiguous)
        stats = lower_bound(y[served], w[served], z)
        if not np.isfinite(stats["ess"]) or stats["ess"] < min_ess:
            continue
        if stats["lcb"] >= target:
            possible = certain | ambiguous
            best = dict(lam=float(lam), **stats, mean_possible=float((w[possible] * y[possible]).sum() / w[possible].sum()),
                        n_ambiguous=int(ambiguous.sum()))
        else:
            break
    return best
