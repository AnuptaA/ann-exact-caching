"""E10: score intervals for window queries kept at 4, 8 and 16 bits.

Linear model on the three v11 features (exact searches, the 32 nearest centers), trained on the training role at
full precision. Calibration and held-out queries are quantized; each gets a score interval from its stored error
(recal.micro.intervals). Lambda is certified on the calibration role with the worst served set the intervals allow
and applied to the full-precision scores, as serving would; compared with full-precision certification.
"""

import numpy as np

from recal import policy
from recal.micro import e3, fit, intervals as iv, world as w

NAME = "E10"
BITS = (4, 8, 16)
MAX_QUERIES = 10_000
NEAREST = 32
CURVE_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)


def context(world: w.World) -> dict:
    """Features, model, labels and thresholds shared by E10, E11 and E12 (cached on the world)."""
    if "v11" in world.cache:
        return world.cache["v11"]
    feats = iv.Features(world, c=NEAREST)
    hist = world.history()
    train_vec = hist["vec"][hist["role"] == w.ROLE_CODES["train"]]
    F_t = feats.exact(train_vec)
    has_t, y_t = feats.candidate_recall(train_vec, e3.role_truth(world, "train"))
    ok = has_t & np.isfinite(F_t).all(axis=1)
    model = fit.fit_linear(F_t[ok], y_t[ok]) if ok.sum() >= 10 else None
    ctx = dict(feats=feats, model=model)
    if model is not None:
        train_scores = fit.score(model, F_t[ok])
        ctx.update(train_scores=train_scores, edges=policy.threshold_edges(train_scores),
                   cal=labeled_set(world, feats, model, *role_queries(world, "cal")),
                   held=labeled_set(world, feats, model, *held_queries(world)))
    world.cache["v11"] = ctx
    return ctx


def role_queries(world: w.World, name: str):
    _, vec = world.role_rows(name)
    return vec, e3.role_truth(world, name), np.ones(len(vec)), None


def held_queries(world: w.World):
    held = world.heldout()
    vec, truth = held["vec"], held["truth"]
    weight = np.bincount(held["inverse"], minlength=len(vec)).astype(np.float64)
    keep = np.arange(len(vec))
    if len(vec) > MAX_QUERIES:
        keep = np.linspace(0, len(vec) - 1, MAX_QUERIES).astype(int)
        vec, truth, weight = vec[keep], truth[keep], weight[keep]
    return vec, truth, weight, keep


def labeled_set(world, feats, model, vec, truth, weight, keep=None) -> dict:
    F = feats.exact(vec)
    has, y = feats.candidate_recall(vec, truth)
    has &= np.isfinite(F).all(axis=1)
    score = np.full(len(vec), -np.inf)
    score[has] = fit.score(model, F[has])
    return dict(vec=vec, F=F, has=has, y=y, score=score, weight=weight, keep=keep)


def weighted_rate(mask: np.ndarray, weight: np.ndarray) -> float:
    return float((mask * weight).sum() / weight.sum()) if weight.sum() else float("nan")


def weighted_mean(values: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> float:
    return float((values[mask] * weight[mask]).sum() / weight[mask].sum()) if weight[mask].sum() > 0 else float("nan")


def intervals_of(ctx: dict, data: dict, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qhat, err = iv.quantize_with_error(data["vec"], bits)
    lower, upper = ctx["feats"].bounds(qhat, err)
    low, high = iv.score_interval(ctx["model"], lower, upper)
    return low, high, err


def violations(data: dict, low: np.ndarray, high: np.ndarray) -> int:
    s = data["score"][data["has"]]
    return int(((s < low[data["has"]] - iv.TOLERANCE) | (s > high[data["has"]] + iv.TOLERANCE)).sum())


def run(world: w.World) -> list[dict]:
    ctx = context(world)
    if ctx["model"] is None or not ctx["cal"]["has"].any():
        return []
    p, cal, held = world.p, ctx["cal"], ctx["held"]
    edges = ctx["edges"]
    ok = cal["has"]
    full = fit.certify(ctx["train_scores"], cal["score"][ok], cal["y"][ok], p.T, p.delta, p.min_ess)
    hit_full = fit.decide(held["score"], held["has"], full.lam)
    rows = []
    for bits in BITS:
        c_low, c_high, c_err = intervals_of(ctx, cal, bits)
        h_low, h_high, h_err = intervals_of(ctx, held, bits)
        cert = iv.certify_worst_case(edges, c_low[ok], c_high[ok], cal["y"][ok], p.T, p.delta, p.min_ess)
        lam = cert["lam"]
        hit = fit.decide(held["score"], held["has"], lam)
        ambiguous = held["has"] & (h_high >= lam) & (h_low < lam) if lam is not None else np.zeros(len(hit), bool)
        curve = [weighted_rate(held["has"] & (h_high >= q) & (h_low < q), held["weight"])
                 for q in np.quantile(ctx["train_scores"], CURVE_LEVELS)]
        rows.append(dict(
            workload=world.name, budget=world.budget, bits=bits, n_held=len(held["vec"]), n_cal=int(ok.sum()),
            error_mean=float(np.mean(np.r_[c_err, h_err])), lam_full=full.lam, lam_wc=lam, certified=lam is not None,
            ambiguous_share=weighted_rate(ambiguous, held["weight"]) if lam is not None else float("nan"),
            ambiguous_share_curve=curve, hit_rate_full=weighted_rate(hit_full, held["weight"]),
            hit_rate_wc=weighted_rate(hit, held["weight"]) if lam is not None else float("nan"),
            hit_cost=weighted_rate(hit_full, held["weight"]) - weighted_rate(hit, held["weight"])
            if lam is not None else float("nan"),
            recall_of_hits_full=weighted_mean(held["y"], hit_full, held["weight"]),
            recall_of_hits_wc=weighted_mean(held["y"], hit, held["weight"]) if lam is not None else float("nan"),
            violations=violations(cal, c_low, c_high) + violations(held, h_low, h_high)))
    return rows
