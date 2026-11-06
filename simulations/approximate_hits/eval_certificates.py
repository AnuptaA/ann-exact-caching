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

from utils.cache.population import hits_from_histograms, online_hits
from utils.cache.workloads import load_inputs
from utils.base.knn import exact_topk, pairwise_dist, sq_norms
from utils.cache.entries import build_entry_tables
from utils.cache.parallel import worker_pool

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHUNK = 2048

#-------------------------------------------------------------------------------


def load_run(run_dir: Path):
    data = json.loads((run_dir / "results.json").read_text())
    args = argparse.Namespace(**data["config"])
    if "zipf" in data["config"]:
        from utils.cache.workloads import build_inputs
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


def certify_state(entries, profile_sizes, g):
    """Lower-bound counts (retained vectors, 0..k) per test query per method."""
    k = g.k
    db, cand_q, test_q = g.db, g.cand_q, g.test_q
    n_kv = len(g.k_values)
    radius, prof_d, prof_id = g.radius, g.prof_d, g.prof_id
    n_test = len(test_q)

    in_cache = np.zeros(len(db), dtype=bool)
    for j in entries:
        in_cache[g.cached_ids[j]] = True
    cached = np.flatnonzero(in_cache)
    out = {"oracle": in_cache[g.test_topk].sum(axis=1)}
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
    db_c_sq = sq_norms(db_c.astype(np.float64))
    ent_q_sq = sq_norms(ent_q.astype(np.float64))

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


def _policy(counts, target, g):
    k = g.k
    w = g.test_w
    n = int(w.sum())
    hist = np.rint(np.bincount(counts, weights=w, minlength=k + 1)).astype(np.int64)
    h, _ = hits_from_histograms(hist, target, k)
    seq = counts if g.test_seq is None else counts[g.test_seq]
    h_on, _ = online_hits(seq, target, k)
    return int(h) / n, h_on / len(seq)


def eval_target(res, g):
    target = res["target"]
    rows = []
    for p in res["points"]:
        entries = res["selected"][:p["n_entries"]]
        counts = certify_state(entries, g.profile_sizes, g)
        oracle = counts["oracle"]
        row = {"cache_frac": p["cache_frac"], "n_entries": p["n_entries"],
               "stored_oracle_hit_rate": p["test_hit_rate"]}
        n_test = len(oracle)
        for name, c in counts.items():
            if name != "oracle":
                row[f"{name}_violations"] = int((c > oracle).sum())
            best, onl = _policy(c, target, g)
            row[f"{name}_hit_rate"] = best
            row[f"{name}_hit_rate_online"] = onl
            row[f"{name}_mean_bound"] = float(c.mean() / g.k)
        # profile storage overhead in database-vector equivalents
        vec_bytes = 4 * g.db.shape[1]
        for m in g.profile_sizes:
            row[f"a3_{m}_overhead_frac"] = p["n_entries"] * m * 8 / vec_bytes / len(g.db)
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
    g = argparse.Namespace(k=cfg.k, k_values=cfg.k_values, db=db, cand_q=cand_q, test_q=test_q,
                           test_w=test_w, test_seq=test_seq, test_topk=test_topk,
                           cached_ids=cached_ids, radius=radius, prof_d=prof_d,
                           prof_id=prof_id, profile_sizes=args.profile_sizes)

    logger.info(f"Evaluating {len(data['results'])} targets...")
    with worker_pool(eval_target, args.workers, len(data["results"]), g=g) as pool:
        out = pool.map(data["results"])

    path = run_dir / "certificates.json"
    path.write_text(json.dumps({"profile_sizes": args.profile_sizes, "results": out}))
    viol = sum(r[key] for t in out for r in t["rows"] for key in r if key.endswith("_violations"))
    logger.info(f"Wrote {path}; total lower-bound violations: {viol}")


if __name__ == "__main__":
    main()
