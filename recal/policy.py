"""Hit policy (H1 to H11): inverse-probability-weighted labels, a normal lower bound and the lambda search.

Items of the calibration window are (score bin, recall, weight) over the labeled queries of the last W
main-path requests. Running sums per score bin make a recalibration one pass over the bins (H7).
"""

from collections import deque
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


def logistic_model(F: np.ndarray, y: np.ndarray) -> dict:
    """Logistic regression on three standardized features with the recall as a soft label (Q6)."""
    from sklearn.linear_model import LogisticRegression
    mean, scale = F.mean(axis=0), F.std(axis=0) + 1e-9
    Z = (F - mean) / scale
    n = len(y)
    lr = LogisticRegression(C=1.0, max_iter=3000)
    lr.fit(np.vstack([Z, Z]), np.r_[np.ones(n), np.zeros(n)], sample_weight=np.r_[y, 1.0 - y] + 1e-12)
    return {"mean": mean, "scale": scale, "coef": lr.coef_[0].copy(), "intercept": float(lr.intercept_[0])}


def score_features(F: np.ndarray, model: dict) -> np.ndarray:
    """ret(q): the model's probability, as the core computes it."""
    z = model["intercept"] + ((F - model["mean"]) / model["scale"]) @ model["coef"]
    return 1.0 / (1.0 + np.exp(-z))


def threshold_edges(train_scores: np.ndarray, n: int = 100) -> np.ndarray:
    """n quantiles (0, 1/n, ..., (n-1)/n) of the training scores, ascending."""
    return np.quantile(train_scores, np.arange(n) / n)


def window_size(H: int, a: float, h0: float, items: float = 200.0) -> tuple[int, bool]:
    """W = min(H, 200 / (a h0)); the flag is True when a h0 H < 200 (margin cannot be reached)."""
    if a * h0 * H < items:
        return H, True
    return int(min(H, np.ceil(items / (a * h0) - 1e-9))), False


@dataclass
class Certified:
    lam: float | None
    ess: float = 0.0
    mean: float = float("nan")
    lcb: float = float("nan")
    n_labeled: int = 0


class CalibrationWindow:
    def __init__(self, edges: np.ndarray, W: int, target: float, delta: float, min_ess: float = 100.0):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.nb = len(self.edges)
        self.W, self.target, self.min_ess = W, target, min_ess
        self.z = NormalDist().inv_cdf(1.0 - delta)
        self.sums = np.zeros((4, self.nb + 1))     # slot 0: below every threshold; slot b+1: bin b
        self.items: deque = deque()                # (request index, slot, recall, weight)
        self.since_exact = 0

    def slot_of(self, score: float) -> int:
        return int(np.searchsorted(self.edges, score, side="right"))

    def _apply(self, slot: int, y: float, w: float, sign: float) -> None:
        self.sums[0, slot] += sign * w
        self.sums[1, slot] += sign * w * y
        self.sums[2, slot] += sign * w * y * y
        self.sums[3, slot] += sign * w * w

    def add(self, request: int, score: float, y: float, w: float) -> None:
        slot = self.slot_of(score)
        self.items.append((request, slot, y, w))
        self._apply(slot, y, w, 1.0)
        self.expire(request)

    def expire(self, request: int) -> None:
        """Drop items older than W main-path requests."""
        while self.items and request - self.items[0][0] >= self.W:
            _, slot, y, w = self.items.popleft()
            self._apply(slot, y, w, -1.0)
            self.since_exact += 1
        if self.since_exact >= 50_000:
            self._recompute()

    def _recompute(self) -> None:
        self.sums[:] = 0.0
        for _, slot, y, w in self.items:
            self._apply(slot, y, w, 1.0)
        self.since_exact = 0

    def __len__(self) -> int:
        return len(self.items)

    def audited_recall(self) -> tuple[float, int]:
        """Mean recall of the audited hits (weight above 1) in the window, for the H11 report."""
        ys = [y for _, _, y, w in self.items if w > 1.0]
        return (float(np.mean(ys)) if ys else float("nan")), len(ys)

    def certify(self) -> Certified:
        """Strictest threshold first; skip low-ESS thresholds, stop at the first failure (H4, H5)."""
        # suffix sums over bins nb-1 .. 0 (slot b+1): items scoring at or above edge b
        suffix = np.cumsum(self.sums[:, :0:-1], axis=1)   # column t: bin nb-1-t and above
        W, S1, S2, SW2 = suffix
        best = Certified(None, n_labeled=len(self.items))
        with np.errstate(divide="ignore", invalid="ignore"):
            ess = W * W / SW2
            mean = S1 / W
            denom = W - SW2 / W
            sd = np.sqrt(np.maximum(S2 - W * mean ** 2, 0.0) / denom)
            lcb = mean - self.z * sd / np.sqrt(ess)
        for t in range(self.nb):
            if not np.isfinite(ess[t]) or ess[t] < self.min_ess:
                continue
            if lcb[t] >= self.target:
                best = Certified(float(self.edges[self.nb - 1 - t]), float(ess[t]), float(mean[t]), float(lcb[t]),
                                 len(self.items))
            else:
                break
        return best


class Status:
    """H10: why lambda is undefined, the share of the last 0.2H main-path requests with lambda undefined."""

    def __init__(self, span: int):
        self.span = span
        self.flags: deque = deque()
        self.undefined = 0
        self.last_lambda: float | None = None
        self.margin_unreachable = False

    def record(self, lam: float | None) -> None:
        undefined = lam is None
        self.flags.append(undefined)
        self.undefined += int(undefined)
        if len(self.flags) > self.span:
            self.undefined -= int(self.flags.popleft())
        if lam is not None:
            self.last_lambda = lam

    def share(self) -> float:
        return self.undefined / len(self.flags) if self.flags else 0.0

    def report(self, lam: float | None, n_labeled: int, ess: float, min_ess: float) -> dict:
        if lam is not None:
            reason = "defined"
        elif n_labeled == 0:
            reason = "no labeled queries"
        elif ess < min_ess:
            reason = "effective sample size below the floor"
        else:
            reason = "no threshold certified"
        return {"lambda": lam, "reason": reason, "undefined_share": self.share(), "last_lambda": self.last_lambda,
                "margin_unreachable": self.margin_unreachable}
