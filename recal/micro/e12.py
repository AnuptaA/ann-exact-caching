"""E12: training the linear model without vectors (half A keeps no vectors, so its features are never refreshed).

The training role's features are computed once against the cache of the first half of the history (an earlier
state, labels included) and the model is fit on them. Only training queries that arrived after that history are
used (the earlier ones were that cache's own centers and selection queries); the serving cache is the full-history one. Compared with the
model trained on fresh features of the same queries: Spearman correlation of held-out scores, and hit rate at T
with lambda certified on fresh calibration scores. The threshold grid of the stale model comes from its own
(stale) training scores.
"""

import numpy as np
from scipy.stats import spearmanr

from recal import policy
from recal.micro import e3, e10, fit, intervals as iv, world as w

NAME = "E12"
EARLIER_SHARE = 0.5


def stale_model(world: w.World) -> dict:
    """Model fit on training-role features computed against the earlier cache."""
    args = dict(world.cache["build_args"])
    early = w.build_world(**args, history_frac=world.history_len / len(world.stream) * EARLIER_SHARE)
    feats = iv.Features(early, c=e10.NEAREST)
    hist = world.history()
    is_train = hist["role"] == w.ROLE_CODES["train"]
    # rows before the earlier cache's history end were its own centers and selection queries
    late = np.flatnonzero(is_train) >= early.history_len
    train_vec = hist["vec"][is_train][late]
    F = feats.exact(train_vec)
    has, y = feats.candidate_recall(train_vec, e3.role_truth(world, "train")[late])
    ok = has & np.isfinite(F).all(axis=1)
    if ok.sum() < 10:
        return dict(model=None, entries=early.built.n_entries, pool=len(early.built.pool_ids))
    model = fit.fit_linear(F[ok], y[ok])
    return dict(model=model, edges=policy.threshold_edges(fit.score(model, F[ok])), entries=early.built.n_entries,
                pool=len(early.built.pool_ids), n_train=int(ok.sum()), late=late)


def fresh_on_same_rows(world: w.World, ctx: dict, late: np.ndarray) -> dict | None:
    """Model fit on the same training queries but with features against the serving cache (same sample size)."""
    feats = ctx["feats"]
    hist = world.history()
    train_vec = hist["vec"][hist["role"] == w.ROLE_CODES["train"]][late]
    F = feats.exact(train_vec)
    has, y = feats.candidate_recall(train_vec, e3.role_truth(world, "train")[late])
    ok = has & np.isfinite(F).all(axis=1)
    if ok.sum() < 10:
        return None
    model = fit.fit_linear(F[ok], y[ok])
    return dict(model=model, edges=policy.threshold_edges(fit.score(model, F[ok])))


def evaluate(world: w.World, model: dict, edges: np.ndarray, ctx: dict) -> dict:
    p, cal, held = world.p, ctx["cal"], ctx["held"]
    ok = cal["has"]
    cal_score = fit.score(model, cal["F"][ok])
    cert = fit.certify(edges, cal_score, cal["y"][ok], p.T, p.delta, p.min_ess)
    score = np.full(len(held["vec"]), -np.inf)
    score[held["has"]] = fit.score(model, held["F"][held["has"]])
    hit = fit.decide(score, held["has"], cert.lam)
    return dict(lam=cert.lam, score=score, hit_rate=e10.weighted_rate(hit, held["weight"]),
                recall_of_hits=e10.weighted_mean(held["y"], hit, held["weight"]))


def run(world: w.World) -> list[dict]:
    ctx = e10.context(world)
    if ctx["model"] is None:
        return []
    stale = stale_model(world)
    if stale["model"] is None:
        return []
    fresh = evaluate(world, ctx["model"], ctx["edges"], ctx)
    old = evaluate(world, stale["model"], stale["edges"], ctx)
    same = fresh_on_same_rows(world, ctx, stale["late"])
    same_eval = evaluate(world, same["model"], same["edges"], ctx) if same is not None else None
    has = ctx["held"]["has"]
    rho = float(spearmanr(fresh["score"][has], old["score"][has])[0]) if has.sum() > 2 else float("nan")
    return [dict(workload=world.name, budget=world.budget, n_entries=world.built.n_entries,
                 n_entries_earlier=stale["entries"], pool_vectors=len(world.built.pool_ids),
                 pool_vectors_earlier=stale["pool"], n_train=stale["n_train"], spearman=rho,
                 lam_fresh=fresh["lam"], lam_stale=old["lam"], hit_rate_fresh=fresh["hit_rate"],
                 hit_rate_stale=old["hit_rate"] if old["lam"] is not None else float("nan"),
                 hit_rate_fresh_same_rows=same_eval["hit_rate"] if same_eval is not None and same_eval["lam"]
                 is not None else float("nan"),
                 recall_of_hits_fresh=fresh["recall_of_hits"],
                 recall_of_hits_stale=old["recall_of_hits"] if old["lam"] is not None else float("nan"),
                 coef_fresh=np.asarray(ctx["model"]["coef"]).tolist(),
                 coef_stale=np.asarray(stale["model"]["coef"]).tolist())]
