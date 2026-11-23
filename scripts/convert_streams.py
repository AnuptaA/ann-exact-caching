#!/usr/bin/env python3
"""Turn a shift stream or an update schedule into a prep folder every driver reads (recal/prep_convert.py).

  # shifts: one prep per kind and seed
  python -m scripts.convert_streams shift --stream shift_abrupt_seed0.npz --kind abrupt --official q.npy \\
      --base base.npy --out preps/abrupt_seed0 [--append-zero] [--k-mixed] [--groups 10000 --group-size 50]
  # updates: adds updates.npz and the files the real Aker driver reads to an existing prep
  python -m scripts.convert_streams updates --prep preps/abrupt_seed0 --schedule data_targeted5.npz --base base.npy

Request vectors come from the existing official vectors (SPACEV official queries, or TriviaQA's DPR-space
question vectors) through the group universe; no encoder runs.
"""

import argparse
from pathlib import Path

import numpy as np

from recal import prep_convert
from workloads.q1 import shifts
from workloads.q1.universe import build_universe


def shift_command(a) -> int:
    official = np.load(a.official, mmap_mode="r")
    base = np.load(a.base, mmap_mode="r")
    uni = build_universe(len(official), a.groups, a.group_size, a.seed)
    shift = dict(np.load(a.stream))
    if a.centroids:
        centroids = np.load(a.centroids)
    else:
        rng = np.random.default_rng(shifts.SEED)
        rows = np.sort(rng.choice(len(base), min(a.sample, len(base)), replace=False))
        centroids = shifts.kmeans_centroids(np.asarray(base[rows], dtype=np.float32),
                                            allow_fallback=a.allow_numpy_kmeans)
    k_seq = prep_convert.k_sequence(len(shift["ids"]), a.k, a.k_mixed, a.seed)
    prep_convert.convert_shift(a.out, a.kind, shift, uni, official, base, a.base, centroids, k_seq, a.depth,
                               append_zero=a.append_zero)
    return 0


def updates_command(a) -> int:
    base = np.load(a.base, mmap_mode="r")
    prep_convert.convert_updates(a.prep, dict(np.load(a.schedule)), base, a.depth)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shift")
    s.add_argument("--stream", type=Path, required=True)
    s.add_argument("--kind", choices=["abrupt", "gradual", "correlated"], required=True)
    s.add_argument("--official", type=Path, required=True)
    s.add_argument("--base", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--groups", type=int, default=10_000)
    s.add_argument("--group-size", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--centroids", type=Path, help="k-means centroids of the run (default: faiss k-means on a sample)")
    s.add_argument("--sample", type=int, default=100_000)
    s.add_argument("--depth", type=int, default=500)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--k-mixed", action="store_true")
    s.add_argument("--append-zero", action="store_true", help="inner-product form: a 0 coordinate on every request")
    s.add_argument("--allow-numpy-kmeans", action="store_true", help="tests only")
    u = sub.add_parser("updates")
    u.add_argument("--prep", type=Path, required=True)
    u.add_argument("--schedule", type=Path, required=True)
    u.add_argument("--base", type=Path, required=True)
    u.add_argument("--depth", type=int, default=1000)
    a = ap.parse_args(argv)
    return shift_command(a) if a.command == "shift" else updates_command(a)


if __name__ == "__main__":
    raise SystemExit(main())
