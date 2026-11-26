"""E10: score intervals for window queries kept at 4, 8 and 16 bits (and natively, when the vectors are int8).

Linear model on the three v11 features (exact searches, the 32 nearest centers), trained on the training role at
full precision. Calibration and held-out queries are quantized; each gets a score interval from its stored error
(recal.micro.intervals). Lambda is certified on the calibration role with the worst served set the intervals allow
(a worst-case guarantee) and applied to the full-precision scores, as serving would; compared with full-precision
certification. A probabilistic interval is reported next to it: the score of the copy's own features +- the
ceil(0.99 (n + 1))-th smallest absolute error seen on the n calibration queries, which covers a new exchangeable
query with probability at least 0.99 (split conformal); certification with it is not a worst-case guarantee.
error_ratio_* is |score of the copy - exact score| over the interval half-width. bits "native" is the vectors as
stored (int8 data is exact in 8 bits per coordinate).
"""

import numpy as np

from recal import policy
from recal.micro import e3, fit, intervals as iv, quantize, world as w

NAME = "E10"
BITS = (4, 8, 16)
MAX_QUERIES = 10_000
NEAREST = 32
CURVE_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
PROB_LEVEL = 0.99


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


def interval_detail(ctx: dict, data: dict, bits) -> dict:
    """Worst-case score interval of each query's copy, the error bound e, and the point score from the copy's own
    features (what serving from the copy would compute)."""
    qhat, err = iv.quantize_with_error(data["vec"], bits)
    if bits == "native":
        low = high = point = data["score"]
        return dict(low=low, high=high, err=err, point=point)
    lower, upper = ctx["feats"].bounds(qhat, err)
    low, high = iv.score_interval(ctx["model"], lower, upper)
    F = ctx["feats"].exact(qhat)
    point = np.full(len(F), -np.inf)
    ok = np.isfinite(F).all(axis=1)
    point[ok] = fit.score(ctx["model"], F[ok])
    return dict(low=low, high=high, err=err, point=point)


def intervals_of(ctx: dict, data: dict, bits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = interval_detail(ctx, data, bits)
    return d["low"], d["high"], d["err"]


def conformal_halfwidth(errors: np.ndarray, level: float = PROB_LEVEL) -> float:
    """The ceil(level (n + 1))-th smallest of n observed absolute errors: for a new query exchangeable with those n,
    its error is at most this with probability at least `level` (split conformal); infinite when n is too small."""
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    k = int(np.ceil(level * (len(errors) + 1)))
    return float(errors[k - 1]) if 0 < k <= len(errors) else float("inf")


def error_ratio(exact: np.ndarray, d: dict, has: np.ndarray) -> dict:
    """|score of the copy - exact score| over the interval half-width (queries with a non-zero width)."""
    half = (d["high"] - d["low"]) / 2.0
    ok = has & np.isfinite(half) & (half > 0) & np.isfinite(d["point"])
    ratio = np.abs(d["point"][ok] - exact[ok]) / half[ok]
    pick = lambda q: float(np.percentile(ratio, q)) if len(ratio) else float("nan")  # noqa: E731
    return dict(error_ratio_median=pick(50), error_ratio_p95=pick(95),
                error_ratio_max=float(ratio.max()) if len(ratio) else float("nan"),
                zero_width_share=float((has & ~(half > 0)).sum() / max(has.sum(), 1)))


def violations(data: dict, low: np.ndarray, high: np.ndarray) -> int:
    s = data["score"][data["has"]]
    return int(((s < low[data["has"]] - iv.TOLERANCE) | (s > high[data["has"]] + iv.TOLERANCE)).sum())


def precisions_for(world: w.World) -> tuple:
    """4, 8, 16 bits, plus native when the query vectors are int8-valued (exact in 8 bits per coordinate)."""
    return BITS + (("native",) if quantize.native_info(world.q)["native_bits"] == 8 else ())


def run(world: w.World) -> list[dict]:
    ctx = context(world)
    if ctx["model"] is None or not ctx["cal"]["has"].any():
        return []
    p, cal, held = world.p, ctx["cal"], ctx["held"]
    edges = ctx["edges"]
    ok = cal["has"]
    full = fit.certify(ctx["train_scores"], cal["score"][ok], cal["y"][ok], p.T, p.delta, p.min_ess)
    hit_full = fit.decide(held["score"], held["has"], full.lam)
    info_q, info_db = quantize.native_info(world.q), quantize.native_info(world.db)
    rows = []
    for bits in precisions_for(world):
        dc, dh = interval_detail(ctx, cal, bits), interval_detail(ctx, held, bits)
        cert = iv.certify_worst_case(edges, dc["low"][ok], dc["high"][ok], cal["y"][ok], p.T, p.delta, p.min_ess)
        lam = cert["lam"]
        hit = fit.decide(held["score"], held["has"], lam)
        ambiguous = held["has"] & (dh["high"] >= lam) & (dh["low"] < lam) if lam is not None else \
            np.zeros(len(hit), bool)
        curve = [weighted_rate(held["has"] & (dh["high"] >= q) & (dh["low"] < q), held["weight"])
                 for q in np.quantile(ctx["train_scores"], CURVE_LEVELS)]
        # probabilistic interval: the copy's own score +- a high quantile of the errors seen on the calibration role
        usable = cal["has"] & np.isfinite(dc["point"])
        half_p = conformal_halfwidth(np.abs(dc["point"][usable] - cal["score"][usable])) if bits != "native" else 0.0
        cert_p = iv.certify_worst_case(edges, (dc["point"] - half_p)[ok], (dc["point"] + half_p)[ok], cal["y"][ok],
                                       p.T, p.delta, p.min_ess)
        lam_p = cert_p["lam"]
        hit_p = fit.decide(held["score"], held["has"], lam_p)
        inside = held["has"] & np.isfinite(dh["point"]) & (np.abs(dh["point"] - held["score"]) <= half_p + 1e-12)
        amb_p = held["has"] & (dh["point"] + half_p >= lam_p) & (dh["point"] - half_p < lam_p) \
            if lam_p is not None else np.zeros(len(hit_p), bool)
        wr, wm = weighted_rate, weighted_mean
        rows.append(dict(
            workload=world.name, budget=world.budget, bits=bits, n_held=len(held["vec"]), n_cal=int(ok.sum()),
            error_mean=float(np.mean(np.r_[dc["err"], dh["err"]])), lam_full=full.lam, lam_wc=lam,
            certified=lam is not None, bound_gap=cert["bound_gap"],
            ambiguous_share=wr(ambiguous, held["weight"]) if lam is not None else float("nan"),
            ambiguous_share_curve=curve, hit_rate_full=wr(hit_full, held["weight"]),
            hit_rate_wc=wr(hit, held["weight"]) if lam is not None else float("nan"),
            hit_cost=wr(hit_full, held["weight"]) - wr(hit, held["weight"]) if lam is not None else float("nan"),
            recall_of_hits_full=wm(held["y"], hit_full, held["weight"]),
            recall_of_hits_wc=wm(held["y"], hit, held["weight"]) if lam is not None else float("nan"),
            violations=violations(cal, dc["low"], dc["high"]) + violations(held, dh["low"], dh["high"]),
            **error_ratio(held["score"], dh, held["has"]),
            prob_level=PROB_LEVEL, prob_halfwidth=half_p, prob_n_cal=int(usable.sum()),
            prob_coverage_held=float(inside.sum() / max(held["has"].sum(), 1)), certified_prob=lam_p is not None,
            lam_prob=lam_p, bound_gap_prob=cert_p["bound_gap"],
            ambiguous_share_prob=wr(amb_p, held["weight"]) if lam_p is not None else float("nan"),
            hit_rate_prob=wr(hit_p, held["weight"]) if lam_p is not None else float("nan"),
            hit_cost_prob=wr(hit_full, held["weight"]) - wr(hit_p, held["weight"]) if lam_p is not None
            else float("nan"),
            recall_of_hits_prob=wm(held["y"], hit_p, held["weight"]) if lam_p is not None else float("nan"),
            query_dtype=info_q["dtype"], query_integer_valued=info_q["integer_valued"], query_min=info_q["min"],
            query_max=info_q["max"], native_bits=info_q["native_bits"], db_dtype=info_db["dtype"],
            db_integer_valued=info_db["integer_valued"]))
    return rows
