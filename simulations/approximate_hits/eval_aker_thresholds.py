#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Aker's density-aware per-entry thresholds on our offline-selected caches.

For each saved trace run (run_e2e_oracle_trace.py) and cache checkpoint of the
greedy cache for --target, replay requests in time order with Aker's hit rule
(Aker paper, Sec. 4.3-4.4) instead of our calibrated admission:

  representative  one per cached query (largest selected K for that query);
                  its neighbor list (NRL) is that query's cached top-K
  equality hit    request vector identical to a representative's query
  alias hit       request vector that earlier got an approximate hit; served
                  by the same representative without a threshold check
  approx hit      d(q, nearest representative query) < tau of that entry
  tau update      tau starts at minD / 4 (--tau-init; minD: distance from the cached query
                  to its nearest neighbor); equality hit: tau *= alpha_L,
                  approximate hit: tau *= alpha_T; tau <= minD
  alias update    which update an alias hit triggers (--alias-update)

The cache content is fixed (no insertion on miss, no eviction), so the only
difference from our method is the hit rule. Two warm-up modes:
  cold  replay the test requests only (taus start at minD / 4)
  warm  replay fit + dev + cal first, report on test only

Retention of a hit (vs exact top-k, as everywhere else):
  union  best answer from all cached vectors (our hit semantics)
  entry  top-k of the representative's NRL by distance to q (Aker's answer)

Writes <run>/aker_thresholds[_<tag>].json. Ceiling variant: --tau-init 1
--alpha-t 1 --alpha-l 1 (static rule d < minD, the loosest Aker allows).
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.eval_certificates import pairwise_dist, _sq_norms
from simulations.approximate_hits.run_e2e_oracle_trace import AKER, WORKLOADS, SPLITS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHUNK = 4096
WARM_SPLITS = ("fit", "dev", "cal")

#-------------------------------------------------------------------------------


def load_run(run_dir: Path):
    """Saved run: config, greedy results, requests and exact neighbors."""
    data = json.loads((run_dir / "results.json").read_text())
    cfg = data["config"]
    z = np.load(run_dir / "topk.npz")
    bounds = {s: tuple(int(x) for x in z[f"bounds_{s}"]) for s in SPLITS}
    lo, hi = bounds["cand"]
    cand_gid = np.unique(z["inv"][lo:hi])  # as split_view: sorted distinct ids
    db = np.load(AKER / WORKLOADS[cfg["trace"]]["base"], mmap_mode="r")
    return {"data": data, "cfg": cfg, "uniq": z["uniq"], "inv": z["inv"],
            "topk": z["topk"][:, :cfg["k"]], "cand_top": z["cand_top"],
            "bounds": bounds, "cand_gid": cand_gid, "db": db}


def fetch(db, ids):
    """Rows of db (possibly memory-mapped) for ids, in ids order, float64."""
    ids = np.asarray(ids, dtype=np.int64)
    order = np.argsort(ids)
    rows = np.empty((len(ids), db.shape[1]))
    rows[order] = db[ids[order]]
    return rows


def representatives(run, entries):
    """One representative per cached query: candidate index and NRL size."""
    n_kv = len(run["cfg"]["k_values"])
    k_of = {}
    for j in entries:
        i, K = j // n_kv, run["cfg"]["k_values"][j % n_kv]
        k_of[i] = max(k_of.get(i, 0), K)
    cand = np.array(sorted(k_of), dtype=np.int64)
    return cand, np.array([k_of[i] for i in cand], dtype=np.int64)


def nearest_rep(qvecs, rep_q, rep_q_sq):
    """Index of and distance to the nearest representative query."""
    idx = np.empty(len(qvecs), dtype=np.int64)
    dist = np.empty(len(qvecs))
    for s in range(0, len(qvecs), CHUNK):
        d = pairwise_dist(qvecs[s:s + CHUNK], rep_q, rep_q_sq)
        idx[s:s + CHUNK] = d.argmin(axis=1)
        dist[s:s + CHUNK] = d[np.arange(len(d)), idx[s:s + CHUNK]]
    return idx, dist


def simulate(stream, n_report, rep_of_gid, near, near_d, tau0, max_tau,
             alpha_t, alpha_l, alias_update):
    """Replay requests (global ids); the last n_report are reported.

    Returns per reported request: hit type (0 miss, 1 equality, 2 alias,
    3 approximate) and the representative used (-1 on a miss).
    """
    tau = tau0.copy()
    alias = {}
    kind = np.zeros(n_report, dtype=np.int8)
    used = np.full(n_report, -1, dtype=np.int64)
    start = len(stream) - n_report
    for t, g in enumerate(stream):
        r = rep_of_gid.get(g)
        if r is not None:
            k, f = 1, alpha_l
        elif g in alias:
            r = alias[g]
            k, f = 2, {"approx": alpha_t, "equality": alpha_l, "none": 1.0}[alias_update]
        else:
            r = near[g]
            if near_d[g] >= tau[r]:
                continue
            k, f = 3, alpha_t
            alias[g] = r
        tau[r] = min(tau[r] * f, max_tau[r])
        if t >= start:
            kind[t - start] = k
            used[t - start] = r
    return kind, used


def entry_retention(run, gids, reps, rep_cand, rep_k, nrl_rows):
    """Retention of answering each request from its representative's NRL."""
    k = run["cfg"]["k"]
    out = np.empty(len(gids))
    for n, (g, r) in enumerate(zip(gids, reps)):
        ids = run["cand_top"][rep_cand[r], :rep_k[r]]
        if len(ids) > k:
            d = pairwise_dist(run["uniq"][g:g + 1], nrl_rows[r])[0]
            ids = ids[np.argpartition(d, k - 1)[:k]]
        out[n] = np.isin(run["topk"][g], ids).sum() / k
    return out


def eval_state(run, point, entries, args):
    cfg, db, k = run["cfg"], run["db"], run["cfg"]["k"]
    rep_cand, rep_k = representatives(run, entries)
    rep_gid = run["cand_gid"][rep_cand]
    rep_q = np.asarray(run["uniq"][rep_gid], dtype=np.float64)
    rep_q_sq = _sq_norms(rep_q)
    nn1 = run["cand_top"][rep_cand, 0]
    min_d = np.sqrt(((rep_q - fetch(db, nn1)) ** 2).sum(axis=1))
    rep_of_gid = {int(g): r for r, g in enumerate(rep_gid)}

    in_cache = np.zeros(cfg["n_db"], dtype=bool)
    for i, K in zip(rep_cand, rep_k):
        in_cache[run["cand_top"][i, :K]] = True

    def split_ids(names):
        return np.concatenate([run["inv"][slice(*run["bounds"][s])] for s in names])

    test = split_ids(["test"])
    warm = split_ids(list(WARM_SPLITS) + ["test"])
    gids = np.unique(warm)
    near_l, near_dl = nearest_rep(run["uniq"][gids], rep_q, rep_q_sq)
    near = dict(zip(gids.tolist(), near_l.tolist()))
    near_d = dict(zip(gids.tolist(), near_dl.tolist()))
    y_union = in_cache[run["topk"][test]].mean(axis=1)

    nrl_rows = {}
    out = {"cache_frac": point["cache_frac"], "n_entries": len(entries),
           "n_reps": len(rep_cand), "oracle_hit_rate": point["test_hit_rate"],
           "oracle_hit_rate_online": point["test_hit_rate_online"]}
    for mode, stream in (("cold", test), ("warm", warm)):
        kind, used = simulate(stream.tolist(), len(test), rep_of_gid, near, near_d,
                              args.tau_init * min_d, min_d, args.alpha_t, args.alpha_l,
                              args.alias_update)
        hit = kind > 0
        hit_reps = used[hit]
        for r in np.unique(hit_reps):
            if r not in nrl_rows and rep_k[r] > k:
                nrl_rows[r] = fetch(db, run["cand_top"][rep_cand[r], :rep_k[r]])
        y_entry = np.ones(len(test))
        if hit.any():
            y_entry[hit] = entry_retention(run, test[hit], hit_reps, rep_cand, rep_k,
                                           nrl_rows)
        n_hit = int(hit.sum())
        res = {"hit_rate": n_hit / len(test),
               "equality_hit_rate": float((kind == 1).mean()),
               "alias_hit_rate": float((kind == 2).mean()),
               "approx_hit_rate": float((kind == 3).mean())}
        for name, y in (("union", y_union), ("entry", y_entry)):
            res[f"hit_retention_{name}"] = float(y[hit].mean()) if n_hit else None
            res[f"effective_retention_{name}"] = float(np.where(hit, y, 1.0).mean())
        out[mode] = res
    return out


#-------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--target", type=float, default=0.9,
                    help="which greedy cache (selected for this target) to use")
    ap.add_argument("--tau-init", type=float, default=0.25, help="initial tau / minD")
    ap.add_argument("--alpha-t", type=float, default=0.25)
    ap.add_argument("--alpha-l", type=float, default=1.1)
    ap.add_argument("--alias-update", choices=["approx", "equality", "none"],
                    default="approx")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    name = "aker_thresholds" + (f"_{args.tag}" if args.tag else "") + ".json"
    for run_dir in map(Path, args.run_dirs):
        logger.info(f"{run_dir.name}: loading...")
        run = load_run(run_dir)
        res = next(r for r in run["data"]["results"] if abs(r["target"] - args.target) < 1e-9)
        states = []
        for p in sorted(res["points"], key=lambda p: p["cache_frac"]):
            entries = res["selected"][:p["n_entries"]]
            if not entries:
                continue
            s = eval_state(run, p, entries, args)
            logger.info(f"  cache {100 * s['cache_frac']:.2f}%: oracle {s['oracle_hit_rate']:.3f}"
                        + "".join(f" | {m} hit {s[m]['hit_rate']:.3f}"
                                  f" (eq {s[m]['equality_hit_rate']:.3f})"
                                  f" ret union {s[m]['hit_retention_union'] or 0:.3f}"
                                  f" entry {s[m]['hit_retention_entry'] or 0:.3f}"
                                  for m in ("cold", "warm")))
            states.append(s)
        (run_dir / name).write_text(json.dumps({"args": vars(args), "states": states},
                                               indent=1))
        logger.info(f"  wrote {run_dir / name}")


if __name__ == "__main__":
    main()
