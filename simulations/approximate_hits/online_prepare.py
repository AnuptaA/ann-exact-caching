#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Inputs for the online simulator (online_sim.py), fixed database: ESCI 1M
(every real query once) or an Aker / SPACEV1B trace (run_e2e_oracle_trace
workloads; repeated requests).

Writes into --out:
  meta.json     dataset description (read by online_sim.load)
  q.npy         distinct query vectors (traces only; ESCI reads its own file)
  top500.npy    exact top-R database ids for every distinct query (ground
                truth and the backend's answers; R = --R)
  clusters.npy  k-means cluster of every distinct query (--n-clusters)
  streams.npz   request orders (indices into the distinct queries):
                  random  all requests in random order (no shift)
                  shift   clusters split into two groups of about equal
                          request counts; group A's requests, then group B's
                          (ESCI: random order within a group; traces: original
                          order); phase_shift = 0 / 1 per position
                  trace   original trace order (traces only); phase_trace for
                          traces made of two parts (spacev-real: log, then
                          official queries)
                  multishift  start mix, mild shift, full shift, return to
                          the start mix (multishift_stream; added with
                          --add-multishift); phase_multishift = 0..3
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ESCI = Path("datasets/esci/data_vectors")

#-------------------------------------------------------------------------------


def split_clusters(labels, rng):
    """Two groups of clusters with about equal query counts (largest first,
    each cluster to the currently smaller group)."""
    sizes = np.bincount(labels)
    group = np.zeros(len(sizes), dtype=np.int64)
    tot = [0, 0]
    for c in np.argsort(-sizes, kind="stable"):
        g = int(tot[1] < tot[0])
        group[c] = g
        tot[g] += sizes[c]
    return group


def shift_stream(labels, group, rng):
    qg = group[labels]
    a, b = np.flatnonzero(qg == 0), np.flatnonzero(qg == 1)
    order = np.r_[rng.permutation(a), rng.permutation(b)]
    phase = np.r_[np.zeros(len(a), dtype=np.int8), np.ones(len(b), dtype=np.int8)]
    return order, phase


def write_streams(out, labels, requests, rng, natural_phase=None):
    """random / shift / trace (original order; its natural phase if given)
    streams over request sequences (indices into the distinct query
    vectors). natural_phase=False: no trace stream (ESCI)."""
    group = split_clusters(labels, rng)
    qg = group[labels[requests]]
    a, b = np.flatnonzero(qg == 0), np.flatnonzero(qg == 1)
    shift = np.r_[requests[a], requests[b]]                # original order within halves
    phase = np.r_[np.zeros(len(a), np.int8), np.ones(len(b), np.int8)]
    extra = {} if natural_phase is False else {"trace": requests}
    if natural_phase is not None and natural_phase is not False:
        extra["phase_trace"] = natural_phase
    np.savez(out / "streams.npz", random=rng.permutation(requests), shift=shift,
             phase_shift=phase, phase=phase, group=group, **extra)
    logger.info(f"shift stream: phase A {len(a)}, B {len(b)} requests")


def _mild_sets(sizes, a_cl, b_cl, mild):
    """Small A clusters holding about `mild` of A's queries (left out in the
    mild phase) and small B clusters (the new ones in the mild phase)."""
    drop, acc = [], 0
    for c in a_cl[np.argsort(sizes[a_cl], kind="stable")]:
        if acc + sizes[c] > mild * sizes[a_cl].sum():
            break
        drop.append(c)
        acc += sizes[c]
    add, acc = [], 0
    for c in b_cl[np.argsort(sizes[b_cl], kind="stable")]:
        add.append(c)
        acc += sizes[c]
        if acc >= 2 * mild * sizes[b_cl].sum() / (1 + mild):
            break
    return np.array(drop, dtype=np.int64), np.array(add, dtype=np.int64)


def _phase_counts(sizes, group, drop, add, start, mild):
    """Per-cluster draw counts (4 phases x clusters) for the largest feasible
    phase length L: start mix = A clusters in proportion to their sizes;
    mild = the kept A clusters in start proportions (share 1 - mild) plus the
    added B clusters in proportion to their sizes (share mild); full = B in
    proportion to sizes; return = start mix."""
    a = (group == 0).astype(float)
    pa = sizes * a / (sizes * a).sum()
    keep = pa.copy()
    keep[drop] = 0
    keep /= keep.sum()
    pb = sizes * (1 - a) / (sizes * (1 - a)).sum()
    padd = np.zeros(len(sizes))
    padd[add] = sizes[add] / sizes[add].sum()
    per_L = (1 - mild) * keep + mild * padd + pb + pa      # demand per unit L
    room = sizes - pa * start
    if (room < 0).any():
        return None, 0
    L = int(np.floor(np.min(room[per_L > 0] / per_L[per_L > 0])))
    counts = np.stack([np.floor(pa * start), np.floor((1 - mild) * L * keep),
                       np.floor(mild * L * padd), np.floor(L * pb), np.floor(L * pa)])
    return counts.astype(np.int64), L


def multishift_stream(labels, requests, rng, warmup=20000, stable=5000, mild=0.2,
                      keep_order=False):
    """Stream with three shifts after a stable start (positions into
    requests; each request used at most once):
      phase 0  warm-up + stable queries, start mix (group A clusters)
      phase 1  mild shift: clusters of A holding about `mild` of its queries
               left out, a share `mild` of the queries from a few B clusters
      phase 2  full shift: group B only
      phase 3  return to the start mix
    Every phase draws exact per-cluster counts (start mix = A clusters in
    proportion to their sizes, so phases 0 and 3 have the same mix); phases
    1-3 have about equal length L, the largest feasible one, and the
    clusters are split into A / B (A larger: it supplies phases 0, 1, 3) to
    maximize L. Within a phase: random order (ESCI) or original order
    (traces, keep_order).
    Returns order (request positions), phase (0..3), group, mild sets, L."""
    lab = labels[requests]
    sizes = np.bincount(lab, minlength=labels.max() + 1).astype(float)
    start = warmup + stable
    best = None
    for frac in np.linspace(0.5, 0.85, 36):
        group = np.ones(len(sizes), dtype=np.int64)
        tot = 0
        for c in np.argsort(-sizes, kind="stable"):
            if tot == 0 or tot + sizes[c] <= frac * len(lab):
                group[c] = 0
                tot += sizes[c]
        a_cl, b_cl = np.flatnonzero(group == 0), np.flatnonzero(group == 1)
        if not len(b_cl):
            continue
        drop, add = _mild_sets(sizes, a_cl, b_cl, mild)
        counts, L = _phase_counts(sizes, group, drop, add, start, mild)
        if counts is not None and (best is None or L > best[-1]):
            best = (group, drop, add, counts, L)
    group, drop, add, counts, L = best
    left = np.ones(len(lab), dtype=bool)
    order, phase = [], []
    for i, rows in enumerate([[0], [1, 2], [3], [4]]):
        pos = []
        for r in rows:
            for c in np.flatnonzero(counts[r]):
                cand = np.flatnonzero(left & (lab == c))
                pick = rng.choice(cand, counts[r, c], replace=False)
                left[pick] = False
                pos.append(pick)
        pos = np.concatenate(pos)
        pos = np.sort(pos) if keep_order else rng.permutation(pos)
        order.append(pos)
        phase.append(np.full(len(pos), i, dtype=np.int8))
    return (np.concatenate(order), np.concatenate(phase), group,
            {"drop": drop.tolist(), "add": add.tolist()}, L)


def add_multishift(out, rng, **kw):
    """Add multishift / phase_multishift / group_multishift to an existing
    streams.npz (written atomically: running jobs keep the old file)."""
    import os
    labels = np.load(out / "clusters.npy")
    z = np.load(out / "streams.npz")
    data = {k: z[k] for k in z.files}
    trace = "trace" in data
    # ESCI: every query once (random arrival order); traces: original order
    requests = data["trace"] if trace else np.arange(len(labels))
    order, phase, group, mild, _ = multishift_stream(labels, requests, rng, keep_order=trace, **kw)
    data.update(multishift=requests[order], phase_multishift=phase, group_multishift=group)
    tmp = out / "streams_tmp.npz"
    np.savez(tmp, **data)
    os.replace(tmp, out / "streams.npz")
    logger.info(f"multishift: phase lengths {np.bincount(phase).tolist()}, mild {mild}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", type=str, default="simulations/approximate_hits/results/online/esci1m")
    ap.add_argument("--dataset", choices=["esci", "trace"], default="esci")
    ap.add_argument("--trace", type=str, default=None,
                    help="run_e2e_oracle_trace workload name, e.g. spacev-aker-0.6")
    ap.add_argument("--n-db", type=int, default=1000000)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add-multishift", action="store_true",
                    help="only add the multishift stream to an existing --out")
    args = ap.parse_args()
    out = Path(args.out)
    if args.add_multishift:
        add_multishift(out, np.random.default_rng(args.seed + 1))
        return
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.dataset == "esci":
        q = np.load(ESCI / "esci_queries.npy").astype(np.float32)
        # each query once; random arrival order (also within the shift halves)
        requests, natural = rng.permutation(len(q)), False
        meta = {"kind": "esci", "n_db": args.n_db}
    else:
        from simulations.approximate_hits.run_e2e_oracle_trace import WORKLOADS, build_trace
        uniq, inv, bounds = build_trace(args.trace)
        q = uniq.astype(np.float32)
        np.save(out / "q.npy", q)
        requests = inv.astype(np.int64)
        parts = WORKLOADS[args.trace]["trace"]
        natural = None
        if len(parts) > 1:      # e.g. spacev-real: query log, then official queries
            n_first = len(_load_first(args.trace))
            natural = (np.arange(len(requests)) >= n_first).astype(np.int8)
        meta = {"kind": "trace", "trace": args.trace}
    (out / "meta.json").write_text(json.dumps(meta))

    if not (out / "streams.npz").exists():
        from sklearn.cluster import MiniBatchKMeans
        logger.info(f"k-means ({args.n_clusters} clusters) on {len(q)} distinct queries...")
        labels = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=0, n_init=3,
                                 batch_size=4096).fit(q).labels_.astype(np.int64)
        np.save(out / "clusters.npy", labels)
        write_streams(out, labels, requests, rng, natural)

    if not (out / f"top{args.R}.npy").exists():
        if args.dataset == "esci":
            db = np.load(ESCI / f"esci_db{args.n_db}_s{args.seed}.npy").astype(np.float32)
            logger.info(f"exact top-{args.R} for {len(q)} queries over {len(db)} vectors...")
            ids = exact_topk(q, db, args.R)
        else:
            from simulations.approximate_hits.run_e2e_oracle_trace import load_base
            from simulations.approximate_hits.run_e2e_oracle_sift import exact_topk_blocked
            db = load_base(args.trace)
            logger.info(f"exact top-{args.R} for {len(q)} queries over {len(db)} vectors...")
            ids = exact_topk_blocked(uniq, db, args.R)
        np.save(out / f"top{args.R}.npy", ids.astype(np.int32))
    logger.info(f"wrote {out}")


def _load_first(trace):
    from simulations.approximate_hits.run_e2e_oracle_trace import AKER, WORKLOADS, _load_vectors
    return _load_vectors(AKER / WORKLOADS[trace]["trace"][0])


if __name__ == "__main__":
    main()
