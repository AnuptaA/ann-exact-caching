"""Models on the three features, v10's certification on a labeled set, and the oracle prefix."""

import numpy as np

from recal import policy

LOGIT = "logit"
LINEAR = "linear"


def fit_logit(F: np.ndarray, y: np.ndarray) -> dict:
    """The current model: fractional logit (logistic regression with the recall as soft label)."""
    return dict(policy.logistic_model(F, y), kind=LOGIT)


def fit_linear(F: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares on recall over the standardized features."""
    mean, scale = F.mean(axis=0), F.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(F)), (F - mean) / scale])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return {"mean": mean, "scale": scale, "coef": beta[1:], "intercept": float(beta[0]), "kind": LINEAR}


def fit(kind: str, F: np.ndarray, y: np.ndarray) -> dict:
    return fit_logit(F, y) if kind == LOGIT else fit_linear(F, y)


def score(model: dict, F: np.ndarray) -> np.ndarray:
    z = model["intercept"] + ((F - model["mean"]) / model["scale"]) @ model["coef"]
    return 1.0 / (1.0 + np.exp(-z)) if model["kind"] == LOGIT else z


def model_json(model: dict) -> dict:
    return {k: (np.asarray(v).tolist() if not isinstance(v, str) else v) for k, v in model.items()}


def certify(edge_scores: np.ndarray, scores: np.ndarray, y: np.ndarray, target: float, delta: float, min_ess: float,
            weights: np.ndarray | None = None) -> policy.Certified:
    """v10's procedure: 100 quantiles of edge_scores as thresholds, labeled items with weights, strictest first."""
    window = policy.CalibrationWindow(policy.threshold_edges(edge_scores), 10**12, target, delta, min_ess)
    w = np.ones(len(scores)) if weights is None else weights
    for s, yy, ww in zip(scores, y, w):
        window.add(0, float(s), float(yy), float(ww))
    return window.certify()


def oracle_count(y: np.ndarray, target: float) -> int:
    """Largest prefix, by decreasing recall, whose mean recall is at least the target."""
    y = np.sort(np.asarray(y, dtype=np.float64))[::-1]
    if not len(y):
        return 0
    ok = np.cumsum(y) / np.arange(1, len(y) + 1) >= target
    return int(np.flatnonzero(ok).max() + 1) if ok.any() else 0


def decide(scores: np.ndarray, has: np.ndarray, lam: float | None) -> np.ndarray:
    """Hit mask: the candidate exists and scores at or above lambda (no lambda: no hit)."""
    if lam is None:
        return np.zeros(len(scores), bool)
    return has & (scores >= lam)
