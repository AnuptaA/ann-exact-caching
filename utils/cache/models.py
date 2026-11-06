#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Retention models."""

import numpy as np

#-------------------------------------------------------------------------------


def gbt_regressor(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=seed)


def fit_ridge(F, y, w, alpha=1.0):
    if np.sum(w) <= 0:
        raise ValueError("weights must have a positive sum")
    mu = np.average(F, axis=0, weights=w)
    sd = np.sqrt(np.average((F - mu) ** 2, axis=0, weights=w)) + 1e-9

    def design(Fx):
        Z = (Fx - mu) / sd
        return np.column_stack([np.ones(len(Z)), Z, Z ** 2])

    X = design(F)
    reg = alpha * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + reg, X.T @ (w * y))
    return lambda Fx: design(Fx) @ beta


def standardize(X, ref):
    mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-9
    return (X - mu) / sd


def logistic_scores(X, y, w):
    """Weighted logistic scores for every split in X (X["fit"] fits and standardizes)."""
    scorer = fit_logistic(X["fit"], y, w, max_iter=2000)
    return {sp: scorer(X[sp]) for sp in X}


def gbt(kind="reg", quantile=None):
    """Gradient-boosted trees: regression ("reg"), classifier ("cls") or quantile loss."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    common = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  early_stopping=True, validation_fraction=0.1, random_state=0)
    if kind == "cls":
        return HistGradientBoostingClassifier(**common)
    if kind == "quantile":
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **common)
    return HistGradientBoostingRegressor(**common)


def fit_logistic(X, y, w=None, max_iter=3000):
    """Logistic regression on standardized X with retention as a soft label.

    Each row is a positive with weight w * y and a negative with weight
    w * (1 - y) (exact cross-entropy on fractional labels). Returns a scorer.
    """
    from sklearn.linear_model import LogisticRegression
    w = np.ones(len(y)) if w is None else w
    if len(y) == 0 or np.sum(w) <= 0:
        raise ValueError("need at least one row with positive weight")
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    n = len(y)
    lr = LogisticRegression(C=1.0, max_iter=max_iter)
    Z = (X - mu) / sd
    lr.fit(np.vstack([Z, Z]), np.r_[np.ones(n), np.zeros(n)],
           sample_weight=np.r_[w * y, w * (1 - y)] + 1e-12)
    return lambda A: lr.decision_function((A - mu) / sd)
