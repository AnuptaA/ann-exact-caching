#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Data-change inputs for the online simulator (context doc 5t): ESCI 1M or a
SPACEV / SPHERE trace (--prep of online_prepare.py --dataset trace).

The query distribution stays fixed (random stream); the database changes:
  churn  hold out a uniform random --frac of the database (initially
         absent); after warm-up insert it, and delete as many uniformly
         random initially present vectors, at the same rate (size and
         distribution stay the same)
  drift  hold out whole database clusters (k-means, --n-db-clusters)
         totaling about --frac, among clusters that a meaningful share of
         queries has true neighbors in; same insert / delete schedule, so
         the only difference from churn is where the inserts land
Schedules: gradual (evenly spread from the end of warm-up to the end of the
stream) and burst (everything at --burst-at).

Ground truth for every database version from one ranking per query over the
union of all vectors: topU<R'>.npy (exact top-R', R' = --R-union); the
answer at time t = the first k ids of the ranking present at t. The ranking
must keep >= R (= 500, shell depth) present ids; the share of queries below
that at the start and the end is reported (simulator flags them).

Writes into --prep: topU<R'>.npy (shared), db_clusters.npy (drift),
data_<kind>.npz (held, initial mask, ins_ids, del_ids, ins_t / del_t for
the gradual schedule, burst_at), rank_extra_<kind>.npz (queries whose union
ranking holds fewer than R vectors present throughout: their ranking merged
with one over the initial database, width 2 R') and data_<kind>.json.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ESCI = Path("datasets/esci/data_vectors")

#-------------------------------------------------------------------------------


def union_ranking(prep, q, db, R):
    path = prep / f"topU{R}.npy"
    if path.exists():
        return np.load(path)
    from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk_blocked
    logger.info(f"exact top-{R} for {len(q)} queries over {len(db)} vectors (union)...")
    ids = exact_topk_blocked(q, db, R, q_chunk=1024).astype(np.int32)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, ids)
    tmp.rename(path)
    return ids


def pick_drift_clusters(labels, top10, frac, rng, min_ratio=0.5):
    """Clusters (random order) whose query share is >= min_ratio x their
    database share, until they hold >= frac of the database. Query share of
    a cluster = share of queries whose true top-10 lies mostly in it."""
    n_c = labels.max() + 1
    db_share = np.bincount(labels, minlength=n_c) / len(labels)
    major = np.array([np.bincount(row, minlength=n_c).argmax() for row in labels[top10]])
    q_share = np.bincount(major, minlength=n_c) / len(major)
    chosen, tot = [], 0.0
    for c in rng.permutation(n_c):
        if q_share[c] >= min_ratio * db_share[c] and db_share[c] > 0:
            chosen.append(int(c))
            tot += db_share[c]
            if tot >= frac:
                break
    return chosen, db_share, q_share


def present_counts(topU, present):
    return present[topU].sum(axis=1)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prep", type=str, default="simulations/approximate_hits/results/online/esci1m")
    ap.add_argument("--kind", choices=["churn", "drift"], required=True)
    ap.add_argument("--n-db", type=int, default=1000000)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--n-db-clusters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20000)
    ap.add_argument("--burst-at", type=int, default=40000)
    ap.add_argument("--R-union", type=int, default=1000)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    prep = Path(args.prep)
    rng = np.random.default_rng(args.seed + (args.kind == "drift"))
    meta_path = prep / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"kind": "esci"}
    if meta["kind"] == "trace":
        # SPACEV / SPHERE traces (online_prepare.py --dataset trace)
        from simulations.approximate_hits.run_e2e_oracle_trace import load_base_f32
        db = load_base_f32(meta["trace"])
        q = np.load(prep / "q.npy").astype(np.float32)
    else:
        db = np.load(ESCI / f"esci_db{args.n_db}_s42.npy").astype(np.float32)
        q = np.load(ESCI / "esci_queries.npy").astype(np.float32)
    n_stream = len(np.load(prep / "streams.npz")["random"])
    topU = union_ranking(prep, q, db, args.R_union)
    info = {"kind": args.kind, "frac": args.frac, "R_union": args.R_union}
    if args.kind == "churn":
        held = rng.choice(len(db), int(args.frac * len(db)), replace=False)
    else:
        path = prep / "db_clusters.npy"
        if path.exists():
            labels = np.load(path)
        else:
            from sklearn.cluster import MiniBatchKMeans
            logger.info(f"k-means ({args.n_db_clusters} clusters) on {len(db)} vectors...")
            labels = MiniBatchKMeans(n_clusters=args.n_db_clusters, random_state=0, n_init=3,
                                     batch_size=8192).fit(db).labels_.astype(np.int64)
            np.save(path, labels)
        chosen, db_share, q_share = pick_drift_clusters(labels, topU[:, :10], args.frac, rng)
        held = np.flatnonzero(np.isin(labels, chosen))
        info.update(clusters=chosen, cluster_db_share=db_share[chosen].tolist(),
                    cluster_query_share=q_share[chosen].tolist())
    initial = np.ones(len(db), dtype=bool)
    initial[held] = False
    ins_ids = rng.permutation(held)
    del_ids = rng.choice(np.flatnonzero(initial), len(held), replace=False)
    n_ev = len(held)
    # gradual: event i applied before the query at position ins_t[i]
    t = args.warmup + (np.arange(n_ev) * (n_stream - args.warmup)) // n_ev
    final = initial.copy()
    final[ins_ids] = True
    final[del_ids] = False
    in_held = np.isin(topU[:, :10], held)
    info.update(n_held=int(n_ev), held_share=n_ev / len(db), n_stream=n_stream,
                queries_with_held_in_top10=float(in_held.any(axis=1).mean()),
                queries_with_5plus_held_in_top10=float((in_held.sum(axis=1) >= 5).mean()))
    for name, pres in (("start", initial), ("end", final)):
        c = present_counts(topU, pres)
        info[f"share_below_R_{name}"] = float((c < args.R).mean())
        info[f"min_present_{name}"] = int(c.min())
    # queries whose union ranking may run out of present vectors at some time
    # (fewer than R vectors that are present throughout): also rank the
    # initial database for them and merge by distance (width 2 R')
    always = initial.copy()
    always[del_ids] = False
    flagged = np.flatnonzero(present_counts(topU, always) < args.R)
    rows = np.full((len(flagged), 2 * args.R_union), -1, dtype=np.int32)
    if len(flagged):
        from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk_blocked
        init_ids = np.flatnonzero(initial)
        logger.info(f"{len(flagged)} queries need the initial-database ranking...")
        topI = init_ids[exact_topk_blocked(q[flagged], db[init_ids], args.R_union,
                                           q_chunk=1024)]
        for i, qi in enumerate(flagged):
            ids = np.unique(np.r_[topU[qi], topI[i]])
            d = ((db[ids].astype(np.float64) - q[qi]) ** 2).sum(axis=1)
            ids = ids[np.argsort(d, kind="stable")][:rows.shape[1]]
            rows[i, :len(ids)] = ids
        ok = (always[np.maximum(rows, 0)] & (rows >= 0)).sum(axis=1)
        info["flagged_min_always_present_after_merge"] = int(ok.min())
    info["n_flagged"] = int(len(flagged))
    np.savez(prep / f"rank_extra_{args.kind}.npz", flagged=flagged, rows=rows)
    np.savez(prep / f"data_{args.kind}.npz", held=held, initial=initial, ins_ids=ins_ids,
             del_ids=del_ids, ins_t=t, del_t=t, burst_at=args.burst_at)
    (prep / f"data_{args.kind}.json").write_text(json.dumps(info, indent=1))
    logger.info(json.dumps(info))


if __name__ == "__main__":
    main()
