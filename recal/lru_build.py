"""First construction of the LRU ablation: replay the history through the cache, then train once."""

import numpy as np

from recal import build as builder
from recal import policy
from recal.build import Built, MODEL_BYTES
from recal.lru import LruCache


def build_lru(snap: dict, backend, p, M: float, request: int, hashes, core, rng) -> tuple[Built, LruCache]:
    d = backend.dim
    vec, K, ids = snap["vec"], snap["k"], snap["ids"]
    R = builder.mini_cache_size(hashes, M, p)
    w_est, _ = policy.window_size(p.H, p.a, 0.2, p.window_items)
    fixed = (p.headroom * M + R * (40.0 + 8.0 * p.K) + w_est * p.item_bytes + 32.0 * (p.n_thresholds + 1)
             + MODEL_BYTES + 8.0 * d)
    roles = builder.split_roles(len(vec), p.roles, rng, vec, p.split_by_query)
    admit = np.sort(np.concatenate([roles["cand"], roles["sel"]]))
    deep = {}
    for Kv in np.unique(K[admit]):
        rows = admit[K[admit] == Kv]
        got_ids, got_d = backend.search(vec[rows], int(4 * Kv), kind="candidate")
        for local, i in enumerate(rows):
            deep[int(i)] = (got_ids[local], got_d[local])
    # shell codes use the range of the cached database vectors, as the core's quantizer does (not the queries')
    lo = np.full(d, np.inf, np.float32)
    hi = np.full(d, -np.inf, np.float32)
    for i in admit:
        held = backend.vectors_of(deep[int(i)][0])
        lo, hi = np.minimum(lo, held.min(axis=0)), np.maximum(hi, held.max(axis=0))
    cache = LruCache(d, M - fixed, p.coord_bytes, lo, hi)
    for i in admit:
        got_ids, got_d = deep[int(i)]
        cache.admit(vec[i], int(K[i]), got_ids, got_d, backend.vectors_of(got_ids))
    Kt = int(p.K)
    train = roles["train"][K[roles["train"]] == Kt]
    cal = roles["cal"][K[roles["cal"]] == Kt]
    needed = cache.entries_needed(vec[train], ids[train][:, :Kt])
    m = builder.choose_m(needed, cache.n_entries(), p.m_percentile, p.idealized)
    cache.model = None

    def features(rows):
        feats, ys = [], []
        for i in rows:
            res = cache.lookup(vec[i], Kt, None, m)
            if res.n_candidates >= Kt:
                feats.append(res.features)
                ys.append(float(np.isin(ids[i, :Kt], res.ids).mean()))
        return np.array(feats).reshape(-1, 3), np.array(ys)

    Ftr, ytr = features(train)
    Fcal, ycal = features(cal)
    if len(ytr) >= 10:
        model = builder.fit_model(Ftr, ytr, p.no_shells)
    else:
        model = {"mean": np.zeros(3), "scale": np.ones(3), "coef": np.zeros(3), "intercept": 0.0}
    cache.set_model(model)
    train_scores = policy.score_features(Ftr, model) if len(ytr) else np.array([0.0])
    edges = policy.threshold_edges(train_scores, p.n_thresholds)
    cscore = policy.score_features(Fcal, model) if len(Fcal) else np.array([])
    window = policy.CalibrationWindow(edges, p.H, p.T, p.delta, p.min_ess)
    for s, yy in zip(cscore, ycal):
        window.add(request, float(s), float(yy), 1.0)
    cert = window.certify()
    h0 = float((cscore >= cert.lam).sum() / len(cscore)) if cert.lam is not None and len(cscore) else 0.0
    W, unreachable = policy.window_size(p.H, p.a, max(h0, 1e-9), p.window_items)
    window.W = W
    window.expire(request)
    core.set_external_bytes("calibration_window", float(W * p.item_bytes + 4 * 8 * (len(edges) + 1)))
    info = {"total_bytes": cache.total_bytes(), "over_budget": False, "entries": cache.n_entries(), "R": R,
            "m": m, "m_source": "history_once", "n_train": int(len(ytr)), "n_cal": int(len(cscore)), "lru": True, "budget": M - fixed}
    built = Built(core, model, m, R, edges, window, cert.lam, cert, h0, W, unreachable, train_scores,
                  np.zeros((0, d), np.float32), np.zeros(0, np.int64), cache.n_entries(), info,
                  np.zeros(0, np.int64), np.zeros(0, np.int64))
    return built, cache
