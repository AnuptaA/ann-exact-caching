"""E3: window vectors kept at 8 or 4 bits, against full precision.

Held-out queries (and the calibration role's) are quantized with a per-vector minimum and scale; features and
scores are computed against the same cache with E1's linear model. Candidate recall stays the true one.
"""

import numpy as np

from recal.micro import e1, fit, world as w
from recal.micro.quantize import quantize

NAME = "E3"
BITS = (8, 4)


def run(world: w.World) -> list[dict]:
    model = e1.fit_models(world)[fit.LINEAR]
    held = world.heldout()
    f, inverse = held["features"], held["inverse"]
    cal, cal_vec = world.role_rows("cal")
    cal_truth = role_truth(world, "cal")
    full_cert = e1.lambda_for(world, model, cal.F, cal.recall, cal.has)
    full_scores = np.where(f.has, fit.score(model, f.F), -np.inf)
    full_hit = fit.decide(full_scores, f.has, full_cert.lam)[inverse]
    y = f.recall[inverse]
    rows = []
    for bits in BITS:
        qf = world.features(quantize(held["vec"], bits), held["truth"])
        q_scores = np.where(qf.has, fit.score(model, qf.F), -np.inf)
        both = f.has & qf.has
        diff = np.abs(q_scores[both] - full_scores[both])
        qc = world.features(quantize(cal_vec, bits), cal_truth)
        q_cert = e1.lambda_for(world, model, qc.F, qc.recall, qc.has)
        q_hit = fit.decide(q_scores, qf.has, full_cert.lam)[inverse]
        mixed_hit = fit.decide(full_scores, f.has, q_cert.lam)[inverse]
        rows.append(dict(
            workload=world.name, budget=world.budget, bits=bits, n_queries=len(held["unique"]),
            score_diff_median=percentile(diff, 50), score_diff_p95=percentile(diff, 95),
            score_diff_p99=percentile(diff, 99), share_with_candidate_in_both=float(both.mean()),
            flip_share=float((full_hit != q_hit).mean()), lam_full=full_cert.lam, lam_quantized=q_cert.lam,
            hit_rate_full=float(full_hit.mean()), hit_rate_quantized_lambda=float(mixed_hit.mean()),
            recall_of_hits_full=mean(y[full_hit]), recall_of_hits_quantized_lambda=mean(y[mixed_hit])))
    return rows


def role_truth(world: w.World, name: str) -> np.ndarray:
    """Exact top-K ids of the history rows of one role, in the order of World.role_rows."""
    sel = world.snap["k"] == world.k
    mask = world.history()["role"] == w.ROLE_CODES[name]
    return world.snap["ids"][sel][mask][:, :world.k]


def percentile(x: np.ndarray, pct: float) -> float:
    return float(np.percentile(x, pct)) if len(x) else float("nan")


def mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")
