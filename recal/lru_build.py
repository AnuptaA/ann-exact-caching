"""First construction of the LRU ablation: replay the history through the cache, then train once."""

import numpy as np

from recal import build as builder
from recal import policy
from recal.build import Built, MODEL_BYTES
from recal.lru import LruCache


def build_lru(snap: dict, backend, p, M: float, request: int, hashes, core) -> tuple[Built, LruCache]:
    d = backend.dim
    vec, K, ids = snap["vec"], snap["k"], snap["ids"]
    R = builder.mini_cache_size(hashes, M, p)
    w_est, _ = policy.window_size(p.H, p.a, 0.2, p.window_items)
    fixed = (p.headroom * M + R * (40.0 + 8.0 * p.K) + w_est * p.item_bytes + 32.0 * (p.n_thresholds + 1)
             + MODEL_BYTES + 8.0 * d)
    cache = LruCache(d, M - fixed, p.coord_bytes, vec.min(axis=0), vec.max(axis=0))
    deep = {}
    for Kv in np.unique(K):
        rows = np.flatnonzero(K == Kv)
        got_ids, got_d = backend.search(vec[rows], int(4 * Kv), kind="candidate")
        for local, i in enumerate(rows):
            deep[int(i)] = (got_ids[local], got_d[local])
    feats, ys = [], []
    for i in range(len(vec)):
        Ki = int(K[i])
        res = cache.lookup(vec[i], Ki, None, p.lru_m)
        if Ki == p.K and res.n_candidates >= Ki:
            feats.append(res.features)
            ys.append(float(np.isin(ids[i, :Ki], res.ids).mean()))
        got_ids, got_d = deep[i]
        cache.admit(vec[i], Ki, got_ids, got_d, backend.vectors_of(got_ids))
    F, y = np.array(feats).reshape(-1, 3), np.array(ys)
    warm = len(F) // 4
    cut = warm + (len(F) - warm) * 2 // 3
    train, cal = slice(warm, cut), slice(cut, len(F))
    if cut - warm >= 10:
        model = builder.fit_model(F[train], y[train], p.no_shells)
    else:
        model = {"mean": np.zeros(3), "scale": np.ones(3), "coef": np.zeros(3), "intercept": 0.0}
    cache.set_model(model)
    train_scores = policy.score_features(F[train], model) if cut > warm else np.array([0.0])
    edges = policy.threshold_edges(train_scores, p.n_thresholds)
    cscore = policy.score_features(F[cal], model) if len(F[cal]) else np.array([])
    window = policy.CalibrationWindow(edges, p.H, p.T, p.delta, p.min_ess)
    for s, yy in zip(cscore, y[cal]):
        window.add(request, float(s), float(yy), 1.0)
    cert = window.certify()
    h0 = float((cscore >= cert.lam).sum() / len(cscore)) if cert.lam is not None and len(cscore) else 0.0
    W, unreachable = policy.window_size(p.H, p.a, max(h0, 1e-9), p.window_items)
    window.W = W
    window.expire(request)
    core.set_external_bytes("calibration_window", float(W * p.item_bytes + 4 * 8 * (len(edges) + 1)))
    info = {"total_bytes": cache.total_bytes(), "over_budget": False, "entries": cache.n_entries(), "R": R,
            "m": p.lru_m, "n_train": int(cut - warm), "n_cal": int(len(cscore)), "lru": True, "budget": M - fixed}
    built = Built(core, model, p.lru_m, R, edges, window, cert.lam, cert, h0, W, unreachable, train_scores,
                  np.zeros((0, d), np.float32), np.zeros(0, np.int64), cache.n_entries(), info,
                  np.zeros(0, np.int64), np.zeros(0, np.int64))
    return built, cache
