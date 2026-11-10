#!/usr/bin/env python3
"""Job lists (one shell command per line) for runs with an inexact backend
(context doc 5aa, 5ab, 9): stable workloads, every method, the backend's
answers used for everything the system sees, retention logged against them
and against exact truth.

Backends: simulated recall (online_sim.py --backend-recall r: each vector
missed independently per query) for every workload; real FAISS indexes
(--backend-file, HNSW / IVF / DiskANN tuned to a target recall@10) where
their answer files exist (ESCI 1M: HNSW, IVF; SPACEV: DiskANN 8-bit). Methods: ours (static, greedy with the score test +
rebuild), LRU (K 10 / 20), seeds 0 and 1; the Aker and QVCache grids of
online_round2_jobs.py on seed 0.
Results: <prep>/bk_<stream>_<backend>_s<seed>/ (backend = r0.95, hnsw0.9, ...).

  python online_backend_jobs.py [--preps ...] > /tmp/backend_jobs.txt
"""

import argparse
from pathlib import Path

from online_round2_jobs import RES, SIM, baselines

WORKLOADS = {"esci1m": ("random", 0.1), "spacev-aker-0.6": ("trace", 0.01),
             "spacev-aker-0.99": ("trace", 0.01), "spacev-aker-0.3": ("trace", 0.01),
             "spacev-real": ("trace", 0.01),
             **{f"triviaqa-{v}-{z}": ("trace", 0.01) for v in ("dpr", "ip")
                for z in ("0.3", "0.6", "0.99")}}
OURS = {"static_default": "--method static",
        "greedy_score_rebuild": "--method greedy --detect score --response rebuild",
        "lru_k10_none": "--method lru --k-lru 10 --detect",
        "lru_k20_none": "--method lru --k-lru 20 --detect"}


def backends(prep, recalls, targets):
    for r in recalls:
        yield f"r{r:g}", f"--backend-recall {r}"
    bdir = Path(RES).parent / "backends" / prep
    for idx in ("hnsw", "ivf", "diskann"):
        if (bdir / f"{idx}.npz").exists():
            for t in targets:
                yield f"{idx}{t:g}", f"--backend-file {bdir}/{idx}.npz --backend-target {t:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preps", nargs="*", default=list(WORKLOADS))
    ap.add_argument("--recalls", type=float, nargs="+", default=[0.95, 0.9])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.9])
    ap.add_argument("--only-index", type=str, default=None, help="only this real index")
    args = ap.parse_args()
    for prep in args.preps:
        stream, budget = WORKLOADS[prep]
        for bname, bflags in backends(prep, [] if args.only_index else args.recalls,
                                      args.targets):
            if args.only_index and not bname.startswith(args.only_index):
                continue
            confs = [(s, n, f) for s in (0, 1) for n, f in OURS.items()] + \
                [(0, n, f) for n, f in baselines()]
            for seed, name, flags in confs:
                method, tag = name.split("_", 1)
                out = f"{RES}/{prep}/bk_{stream}_{bname}_s{seed}"
                print(f"mkdir -p {out}/logs && [ -f {out}/{method}_{tag}.json ] || "
                      f"{SIM} --prep {RES}/{prep} --stream {stream} --budget {budget} "
                      f"--seed {seed} {bflags} {flags} --tag _{tag} --out {out} "
                      f"> {out}/logs/{name}.txt 2>&1")


if __name__ == "__main__":
    main()
