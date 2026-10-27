#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Evaluate provable retention lower bounds (certificates) on saved greedy caches.

For a test query q, the cache answer is the k cached vectors nearest to q; D_k
is the distance to the k-th. An intruder is an uncached data vector closer to q
than D_k; with I intruders, retention >= (k - I) / k. For a cache entry with
query Q, radius r (all data within r of Q are cached), and d = D(q, Q), every
intruder lies within R = D_k + d of Q (triangle inequality). Certificates:

  cig      exact CIG (union): retention = 1 if some entry has R <= r, else 0
  a1       partial CIG: rho = max_entries (r - d); every data vector within rho
           of q is cached, so retention >= min(k, #cached within rho) / k
  a3_M     neighbor-distance profile: each entry also stores the distances and
           ids of Q's next M neighbors beyond its cached ones. Intruders <=
           #(profile neighbors closer than R that are not cached anywhere);
           unbounded if R exceeds the profile. Minimum over entries.
  a1+a3_M  max of the two lower bounds

The caches are the ones greedy selected for ORACLE retention (saved in
results.json); a cache selected for a certificate could do better.
Admission uses each certificate's lower bound in place of true retention, so
the hits-only average target holds for true retention as well.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.greedy_population import hits_from_histograms, online_hits
from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk, load_inputs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHUNK = 2048
_G: dict = {}  # shared inputs for forked workers

#-------------------------------------------------------------------------------


def _sq_norms(x):
    return np.einsum("ij,ij->i", x, x)


def pairwise_dist(a, b, b_sq=None):
    """Euclidean distances between rows of a and rows of b (float64)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if b_sq is None:
        b_sq = _sq_norms(b)
    d2 = _sq_norms(a)[:, None] + b_sq[None, :] - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(d2, 0.0))


def load_run(run_dir: Path):
    data = json.loads((run_dir / "results.json").read_text())
    args = argparse.Namespace(**data["config"])
    if "zipf" in data["config"]:
        from simulations.approximate_hits.run_e2e_oracle_synthetic import build_inputs
        inp = build_inputs(args)
        wl = inp["wl"]
        db = inp["db"]
        cand_q = wl.vectors(inp["cand_ids"])
        test_q = wl.vectors(inp["test_ids"])
        test_w, test_seq = inp["test_w"], inp["test_seq"]
    else:
        db, cand_q, _, test_q = load_inputs(args)
        test_w = np.ones(len(test_q), dtype=np.int64)
        test_seq = None
    return data, args, db, cand_q, test_q, test_w, test_seq


def build_entry_tables(cand_q, db, k_values, m_max, cached_top=None):
    """Per candidate: cached ids (as greedy used them), radius, and profile.

    Cached ids reproduce greedy's candidates exactly (same exact_topk call, or
    the saved neighbors passed as cached_top); the profile is the next nearest
    neighbors excluding the cached ids. With m_max <= 1 the profile is not
    computed (left empty), which is all the calibrated / predictor
    evaluations need.
    """
    k_max = max(k_values)
    if cached_top is None:
        cached_top = exact_topk(cand_q, db, k_max)
    big = exact_topk(cand_q, db, k_max + m_max) if m_max > 1 else None
    n_kv = len(k_values)
    n_cand = len(cand_q) * n_kv
    radius = np.empty(n_cand)
    prof_d = np.full((n_cand, m_max), np.inf)
    prof_id = np.zeros((n_cand, m_max), dtype=np.int64)
    cached_ids = []
    for i in range(len(cand_q)):
        top_ids = cached_top[i]
        d_top = pairwise_dist(cand_q[i:i + 1], db[top_ids])[0]
        if big is not None:
            ids_big = big[i]
            d_big = pairwise_dist(cand_q[i:i + 1], db[ids_big])[0]
        for t, K in enumerate(k_values):
            j = i * n_kv + t
            ids = top_ids[:K]
            cached_ids.append(ids)
            radius[j] = d_top[:K].max()
            if big is not None:
                keep = ~np.isin(ids_big, ids)
                pid = ids_big[keep][:m_max]
                prof_id[j, :len(pid)] = pid
                prof_d[j, :len(pid)] = d_big[keep][:m_max]
    return cached_ids, radius, prof_d, prof_id


def certify_state(entries, profile_sizes):
    """Lower-bound counts (retained vectors, 0..k) per test query per method."""
    k = _G["k"]
    db, cand_q, test_q = _G["db"], _G["cand_q"], _G["test_q"]
    n_kv = len(_G["k_values"])
    radius, prof_d, prof_id = _G["radius"], _G["prof_d"], _G["prof_id"]
    n_test = len(test_q)

    in_cache = np.zeros(len(db), dtype=bool)
    for j in entries:
        in_cache[_G["cached_ids"][j]] = True
    cached = np.flatnonzero(in_cache)
    out = {"oracle": in_cache[_G["test_topk"]].sum(axis=1)}
    methods = ["cig", "a1"] + [f"a3_{m}" for m in profile_sizes] + \
              [f"a1+a3_{m}" for m in profile_sizes]
    for name in methods:
        out[name] = np.zeros(n_test, dtype=np.int64)
    if len(entries) == 0 or len(cached) < k:
        return out

    ent = np.asarray(entries)
    ent_q = cand_q[ent // n_kv]
    ent_r = radius[ent]
    # uncached-profile prefix counts per entry: cum[j, t] = #uncached in first t
    prof_unc = {}
    for m in profile_sizes:
        unc = ~in_cache[prof_id[ent, :m]] & np.isfinite(prof_d[ent, :m])
        cum = np.zeros((len(ent), m + 1), dtype=np.int64)
        np.cumsum(unc, axis=1, out=cum[:, 1:])
        prof_unc[m] = cum
    db_c = db[cached]
    db_c_sq = _sq_norms(db_c.astype(np.float64))
    ent_q_sq = _sq_norms(ent_q.astype(np.float64))

    for s in range(0, n_test, CHUNK):
        e = min(s + CHUNK, n_test)
        q = test_q[s:e]
        d_qc = pairwise_dist(q, db_c, db_c_sq)                 # (c, |C|)
        d_k = np.partition(d_qc, k - 1, axis=1)[:, k - 1]       # k-th cached
        d_qe = pairwise_dist(q, ent_q, ent_q_sq)                # (c, E)
        R = d_k[:, None] + d_qe

        # exact CIG: some entry contains the whole intruder region
        exact = (R <= ent_r[None, :]).any(axis=1)
        out["cig"][s:e] = np.where(exact, k, 0)

        # partial CIG
        rho = (ent_r[None, :] - d_qe).max(axis=1)
        m_cert = (d_qc <= rho[:, None]).sum(axis=1)
        a1 = np.minimum(m_cert, k)
        out["a1"][s:e] = a1

        # profile bound
        for m in profile_sizes:
            best_i = np.full(e - s, np.iinfo(np.int64).max)
            best_i[exact] = 0
            if m > 0:
                cum = prof_unc[m]
                pd = prof_d[ent, :m]
                for jj in range(len(ent)):
                    Rj = R[:, jj]
                    t = np.searchsorted(pd[jj], Rj, side="left")
                    # beyond the stored profile: no bound from this entry
                    covered = (t < m) | (Rj <= ent_r[jj])
                    ij = np.where(Rj <= ent_r[jj], 0, cum[jj, np.minimum(t, m)])
                    best_i = np.where(covered, np.minimum(best_i, ij), best_i)
            a3 = np.where(best_i >= k, 0, k - best_i)
            out[f"a3_{m}"][s:e] = a3
            out[f"a1+a3_{m}"][s:e] = np.maximum(a1, a3)
    return out


def _policy(counts, target):
    k = _G["k"]
    w = _G["test_w"]
    n = int(w.sum())
    hist = np.rint(np.bincount(counts, weights=w, minlength=k + 1)).astype(np.int64)
    h, _ = hits_from_histograms(hist, target, k)
    seq = counts if _G["test_seq"] is None else counts[_G["test_seq"]]
    h_on, _ = online_hits(seq, target, k)
    return int(h) / n, h_on / len(seq)


def eval_target(res):
    target = res["target"]
    rows = []
    for p in res["points"]:
        entries = res["selected"][:p["n_entries"]]
        counts = certify_state(entries, _G["profile_sizes"])
        oracle = counts["oracle"]
        row = {"cache_frac": p["cache_frac"], "n_entries": p["n_entries"],
               "stored_oracle_hit_rate": p["test_hit_rate"]}
        n_test = len(oracle)
        for name, c in counts.items():
            if name != "oracle":
                row[f"{name}_violations"] = int((c > oracle).sum())
            best, onl = _policy(c, target)
            row[f"{name}_hit_rate"] = best
            row[f"{name}_hit_rate_online"] = onl
            row[f"{name}_mean_bound"] = float(c.mean() / _G["k"])
        # profile storage overhead in database-vector equivalents
        vec_bytes = 4 * _G["db"].shape[1]
        for m in _G["profile_sizes"]:
            row[f"a3_{m}_overhead_frac"] = p["n_entries"] * m * 8 / vec_bytes / len(_G["db"])
        rows.append(row)
        assert abs(row["oracle_hit_rate"] - p["test_hit_rate"]) < 1e-9 or n_test == 0, (
            f"oracle mismatch at {p['cache_frac']}: "
            f"{row['oracle_hit_rate']} vs {p['test_hit_rate']}")
    return {"target": target, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--profile-sizes", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    data, cfg, db, cand_q, test_q, test_w, test_seq = load_run(run_dir)
    logger.info(f"{run_dir.name}: db={db.shape} cand={cand_q.shape} test={test_q.shape}")
    logger.info("Building entry tables (cached ids, radius, profiles)...")
    cached_ids, radius, prof_d, prof_id = build_entry_tables(
        cand_q, db, cfg.k_values, max(args.profile_sizes))
    test_topk = exact_topk(test_q, db, cfg.k)
    _G.update(k=cfg.k, k_values=cfg.k_values, db=db, cand_q=cand_q, test_q=test_q,
              test_w=test_w, test_seq=test_seq, test_topk=test_topk,
              cached_ids=cached_ids, radius=radius, prof_d=prof_d, prof_id=prof_id,
              profile_sizes=args.profile_sizes)

    logger.info(f"Evaluating {len(data['results'])} targets...")
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(args.workers, len(data["results"]))) as pool:
            out = pool.map(eval_target, data["results"])
    else:
        out = [eval_target(r) for r in data["results"]]

    path = run_dir / "certificates.json"
    path.write_text(json.dumps({"profile_sizes": args.profile_sizes, "results": out}))
    viol = sum(r[key] for t in out for r in t["rows"] for key in r if key.endswith("_violations"))
    logger.info(f"Wrote {path}; total lower-bound violations: {viol}")


if __name__ == "__main__":
    main()
