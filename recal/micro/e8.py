"""E8: the window against the budget.

A v11 window item keeps an 8-bit vector, 4K ids and 31 bytes: d + 4K + 31. The window is capped at a share of M;
lambda is certified on the calibration half of consecutive stretches of the held-out stream (v10's procedure:
misses weigh 1, audited hits 1/a) and applied to the next stretch. The uncapped reference uses the window the
design asks for without a memory limit, W = 400 / (a h).
"""

import numpy as np

from recal.micro import e1, fit, world as w

NAME = "E8"
CAPS = (0.01, 0.03, 0.05)
FIXED_ITEM_BYTES = 31
UNCAPPED_ITEMS = 400.0


def item_bytes(dim: int, k: int) -> int:
    return dim + 4 * k + FIXED_ITEM_BYTES


def window_for(cap: float | None, M: float, dim: int, k: int, a: float, h: float, n_requests: int) -> dict:
    """Window in main-path requests and labeled items for a memory cap (None: no cap)."""
    labeled = (1.0 - h) + a * h
    if cap is None:
        W = int(np.ceil(UNCAPPED_ITEMS / (a * h))) if h > 0 else n_requests
    else:
        W = int(cap * M // item_bytes(dim, k) / labeled)
    W = max(0, min(W, n_requests))
    return dict(W=W, items=W * labeled, calibration_labeled=W * labeled / 2.0)


def certify_stretches(scores, has, y, edges_scores, W, lam_policy, p, rng) -> list[dict]:
    """Per stretch: lambda certified on its calibration half, then its hit rate on the next stretch."""
    n = len(scores)
    if W < 1:
        return []
    size = min(W, n)
    n_stretch = max(n // size, 1)
    out = []
    for j in range(n_stretch):
        sl = slice(j * size, (j + 1) * size)
        s, hs, yy = scores[sl], has[sl], y[sl]
        hit = fit.decide(s, hs, lam_policy)
        audited = hit & (rng.random(len(s)) < p.a)
        labeled = hs & (~hit | audited)
        half = rng.random(len(s)) < 0.5
        use = labeled & half
        weights = np.where(hit[use], 1.0 / p.a, 1.0)
        cert = fit.certify(edges_scores, s[use], yy[use], p.T, p.delta, p.min_ess, weights) if use.any() \
            else fit.policy.Certified(None)
        if j + 1 < n_stretch:
            nxt = slice((j + 1) * size, (j + 2) * size)
        else:
            nxt = sl if n_stretch == 1 else None
        row = dict(stretch=j, labeled_in_half=int(use.sum()), certified=cert.lam is not None, lam=cert.lam,
                   in_sample=n_stretch == 1)
        if nxt is not None:
            applied = fit.decide(scores[nxt], has[nxt], cert.lam)
            row.update(hit_rate=float(applied.mean()),
                       hits=int(applied.sum()), recall_sum=float(y[nxt][applied].sum()))
        out.append(row)
    return out


def run(world: w.World) -> list[dict]:
    model = e1.fit_models(world)[fit.LOGIT]
    held = world.heldout()
    f, inverse = held["features"], held["inverse"]
    n = len(inverse)
    if model is None or not f.has.any():
        return []
    cal, _ = world.role_rows("cal")
    cert0 = e1.lambda_for(world, model, cal.F, cal.recall, cal.has)
    scores = np.where(f.has, fit.score(model, f.F), -np.inf)[inverse]
    has, y = f.has[inverse], f.recall[inverse]
    h0 = float(fit.decide(scores, has, cert0.lam).mean())
    dim, k, M, p = world.db.shape[1], world.k, world.M, world.p
    rows = []
    for cap in (None, *CAPS):
        win = window_for(cap, M, dim, k, p.a, h0, n)
        rng = np.random.default_rng(p.seed + 17)
        stretches = certify_stretches(scores, has, y, model["train_scores"], win["W"], cert0.lam, p, rng)
        done = [s for s in stretches if "hit_rate" in s]
        hits = sum(s["hits"] for s in done)
        rows.append(dict(
            workload=world.name, budget=world.budget, cap_share_of_M="uncapped" if cap is None else cap,
            M_bytes=M, item_bytes=item_bytes(dim, k), reference_hit_rate=h0, W_requests=win["W"],
            window_items=win["items"], calibration_labeled=win["calibration_labeled"],
            n_stretches=len(stretches), certified_share=float(np.mean([s["certified"] for s in stretches]))
            if stretches else 0.0,
            hit_rate=float(np.mean([s["hit_rate"] for s in done])) if done and any(s["certified"] for s in stretches)
            else float("nan"),
            recall_of_hits=float(sum(s["recall_sum"] for s in done) / hits) if hits else float("nan"),
            share_below_target=float(np.mean([s["recall_sum"] / s["hits"] < p.T for s in done if s["hits"]]))
            if any(s["hits"] for s in done) else float("nan"),
            in_sample=bool(done and done[0]["in_sample"])))
    return rows
