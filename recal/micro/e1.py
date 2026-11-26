"""E1: ordinary least squares against the fractional-logit recall estimator.

Both models are fit on the training role and certified on the calibration role (v10's procedure); the cache is
v10's construction on the history. Hit rate is over the held-out requests (every request is treated as main path).
"""

import numpy as np
from scipy.stats import spearmanr

from recal.micro import fit, world as w

NAME = "E1"
MODELS = (fit.LOGIT, fit.LINEAR)


def fit_models(world: w.World) -> dict:
    """Both models on the training role, with the training scores that give the thresholds."""
    if "models" not in world.cache:
        train, _ = world.role_rows("train")
        ok = train.has
        out = {}
        for kind in MODELS:
            model = fit.fit(kind, train.F[ok], train.recall[ok]) if ok.sum() >= 10 else None
            out[kind] = None if model is None else dict(model, train_scores=fit.score(model, train.F[ok]))
        world.cache["models"] = out
    return world.cache["models"]


def lambda_for(world: w.World, model: dict, F: np.ndarray, y: np.ndarray, has: np.ndarray):
    """Certified lambda from the calibration role's rows (candidates only), as the construction does."""
    ok = has
    return fit.certify(model["train_scores"], fit.score(model, F[ok]), y[ok], world.p.T, world.p.delta,
                       world.p.min_ess)


def evaluate(world: w.World, kind: str) -> dict:
    model = fit_models(world)[kind]
    held = world.heldout()
    f, inverse = held["features"], held["inverse"]
    n = len(inverse)
    if model is None:
        return dict(model=kind, lam=None, certified=False, hit_rate=0.0, recall_of_hits=float("nan"),
                    oracle_hit_rate=0.0, gap=0.0, scores=None)
    cal, _ = world.role_rows("cal")
    cert = lambda_for(world, model, cal.F, cal.recall, cal.has)
    scores = np.where(f.has, fit.score(model, f.F), -np.inf)
    hit = fit.decide(scores, f.has, cert.lam)[inverse]
    y = f.recall[inverse]
    oracle = fit.oracle_count(y[f.has[inverse]], world.p.T) / n
    hit_rate = float(hit.mean())
    return dict(model=kind, lam=cert.lam, certified=cert.lam is not None, hit_rate=hit_rate,
                recall_of_hits=float(y[hit].mean()) if hit.any() else float("nan"), oracle_hit_rate=float(oracle),
                gap=float(oracle - hit_rate), scores=scores)


def run(world: w.World) -> list[dict]:
    """One row per model for this workload and budget."""
    results = {kind: evaluate(world, kind) for kind in MODELS}
    held = world.heldout()
    both = np.isfinite(results[fit.LOGIT]["scores"]) & np.isfinite(results[fit.LINEAR]["scores"]) \
        if results[fit.LOGIT]["scores"] is not None and results[fit.LINEAR]["scores"] is not None else None
    rho = float("nan")
    if both is not None and both.sum() > 2:
        rho = float(spearmanr(results[fit.LOGIT]["scores"][both], results[fit.LINEAR]["scores"][both])[0])
    models = fit_models(world)
    rows = []
    for kind in MODELS:
        r = {k: v for k, v in results[kind].items() if k != "scores"}
        rows.append(dict(workload=world.name, budget=world.budget, **r, spearman=rho,
                         n_requests=len(held["inverse"]), n_entries=world.built.n_entries, m=world.m,
                         coef=None if models[kind] is None else np.asarray(models[kind]["coef"]).tolist(),
                         intercept=None if models[kind] is None else models[kind]["intercept"]))
    return rows


def summarize(rows: list[dict]) -> dict:
    """Largest hit-rate difference between the two models across the workload's budgets."""
    by_budget: dict = {}
    for r in rows:
        by_budget.setdefault(r["budget"], {})[r["model"]] = r["hit_rate"]
    diffs = [abs(v[fit.LOGIT] - v[fit.LINEAR]) for v in by_budget.values() if len(v) == 2]
    return {"max_hit_rate_difference": max(diffs) if diffs else float("nan")}
