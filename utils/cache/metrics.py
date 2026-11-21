#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Hit-rate and retention metrics."""

import numpy as np

#-------------------------------------------------------------------------------


def best_cutoff_rate(s, y, target):
    """Largest fraction admitted by a cutoff on s with mean y >= target."""
    if len(s) == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    mean = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    # only cut between distinct scores
    ok = (mean >= target) & np.append(ss[1:] < ss[:-1], True)
    return float((np.flatnonzero(ok).max() + 1) / len(ys)) if ok.any() else 0.0


def interp_log(xs, ys, x):
    """Interpolate ys in log(x); None outside the measured range."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) == 0 or x <= 0 or xs.min() <= 0:
        return None
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x < xs[0] or x > xs[-1] * 1.001:
        return None
    return float(np.interp(np.log(x), np.log(xs), ys))


def expand_index(w):
    """Row index repeated once per request: row i appears w[i] times."""
    w = np.asarray(w)
    if (w < 0).any() or (w != np.floor(w)).any():
        raise ValueError("weights must be non-negative integers")
    return np.repeat(np.arange(len(w)), w.astype(np.int64))


def admit(lam, s, y):
    """(hit rate, mean retention of the hits or None) when admitting s >= lam."""
    adm = s >= lam if lam is not None else np.zeros(len(s), dtype=bool)
    return float(adm.mean()), (float(y[adm].mean()) if adm.any() else None)


def result_for_target(data, target):
    """The greedy result of a saved run for one retention target."""
    for res in data["results"]:
        if abs(res["target"] - target) < 1e-9:
            return res
    raise ValueError(f"target {target} not in this run; available: "
                     f"{[r['target'] for r in data['results']]}")


def nearest_point(res, size):
    """The checkpoint of a greedy result whose cache fraction is closest to size."""
    if not res["points"]:
        raise ValueError("greedy result has no checkpoints")
    return min(res["points"], key=lambda p: abs(p["cache_frac"] - size))


def state_tasks(res, sizes, target=None):
    """(target, point, entries) per requested cache size, as the evaluators take them."""
    target = res["target"] if target is None else target
    out = []
    for size in sizes:
        p = nearest_point(res, size)
        out.append((target, p, res["selected"][:p["n_entries"]]))
    return out

