#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Joint bits + train-size sweep on real SIFT (data-following train + test, 50k db).

Tests the claim that more training data only helps when PAIRED with finer LSH
buckets: finer buckets reduce smoothing bias but need more train data to stay
populated. We sweep (n_train, n_bits) together and report, for each cell, RMSE,
skill vs the global-average baseline, per-decile bias, and the mean number of
train queries per populated bucket (to see starvation set in).
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from datasets.dataloaders import load_fvecs_fast

from simulations.radius_estimation.lsh_radius_estimator import true_radius

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SIFT1M_DIR = Path("datasets/sift/raw_sift1m/sift")

#-------------------------------------------------------------------------------


def buckets_lsh(train_q, test_q, n_bits, dim, seed):
    rng = np.random.default_rng(seed)
    planes = rng.standard_normal((dim, n_bits))
    pow2 = (1 << np.arange(n_bits)).astype(np.int64)
    tb = ((train_q @ planes) > 0).astype(np.int64) @ pow2
    te = ((test_q @ planes) > 0).astype(np.int64) @ pow2
    return tb, te


def estimate_by_bucket(train_bucket, test_bucket, r_train, global_mean):
    sums, counts = {}, {}
    for b, r in zip(train_bucket.tolist(), r_train.tolist()):
        sums[b] = sums.get(b, 0.0) + r
        counts[b] = counts.get(b, 0) + 1
    est = np.empty(len(test_bucket))
    hit = np.zeros(len(test_bucket), dtype=bool)
    for i, b in enumerate(test_bucket.tolist()):
        if counts.get(b, 0) > 0:
            est[i] = sums[b] / counts[b]
            hit[i] = True
        else:
            est[i] = global_mean
    mean_occ = float(np.mean(list(counts.values()))) if counts else 0.0
    return est, hit, mean_occ


def _metrics(estimate, r_test, global_mean):
    e = estimate - r_test
    rmse = float(np.sqrt(np.mean(e ** 2)))
    rmse_g = float(np.sqrt(np.mean((global_mean - r_test) ** 2)))
    order = np.argsort(r_test)
    es = e[order]
    nlo = len(es) // 10
    return {
        "rmse": rmse,
        "skill": float(1 - rmse / rmse_g) if rmse_g > 0 else float("nan"),
        "mean_error": float(np.mean(e)),
        "std_comp": float(np.std(estimate) / np.std(r_test)),
        "bias_lo": float(np.mean(es[:nlo])) if nlo else float("nan"),
        "bias_hi": float(np.mean(es[-nlo:])) if nlo else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-db", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--trainsizes", type=int, nargs="+",
                    default=[2000, 8000, 24000])
    ap.add_argument("--bits-list", type=int, nargs="+",
                    default=[12, 16, 20, 24])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    logger.info("Loading SIFT1M...")
    base = load_fvecs_fast(str(SIFT1M_DIR / "sift_base.fvecs"), max_n=max(args.n_db * 2, 100000))
    maxtrain = max(args.trainsizes)
    learn = load_fvecs_fast(str(SIFT1M_DIR / "sift_learn.fvecs"), max_n=maxtrain + args.n_test + 2000)

    db_idx = rng.choice(len(base), size=args.n_db, replace=False)
    sift_db = base[db_idx]
    q_idx = rng.choice(len(learn), size=args.n_test + maxtrain, replace=False)
    test_q = learn[q_idx[:args.n_test]]
    train_pool = learn[q_idx[args.n_test:]]

    logger.info("Computing true radii...")
    r_test = true_radius(test_q, sift_db, args.k)
    r_train_pool = true_radius(train_pool, sift_db, args.k)
    logger.info(f"  true radius mean={np.mean(r_test):.1f} std={np.std(r_test):.1f}")

    results = {"config": vars(args), "cells": []}

    print(f"\n{'='*94}")
    print("JOINT (n_train x bits) SWEEP: skill vs global-average baseline")
    print(f"{'='*94}")
    header = "  n_train \\ bits " + "".join(f"{b:>12}" for b in args.bits_list)
    print(header)
    # we print skill grid, then detail table
    grid = {}
    detail = []
    for nt in args.trainsizes:
        tr = train_pool[:nt]
        r_tr = r_train_pool[:nt]
        gm = float(np.mean(r_tr))
        row = []
        for bits in args.bits_list:
            tb, te = buckets_lsh(tr, test_q, bits, sift_db.shape[1], args.seed)
            est, hit, occ = estimate_by_bucket(tb, te, r_tr, gm)
            m = _metrics(est, r_test, gm)
            m.update({"n_train": nt, "bits": bits,
                      "mean_occupancy": occ,
                      "hit_frac": float(np.mean(hit))})
            results["cells"].append(m)
            grid[(nt, bits)] = m
            row.append(m["skill"])
            detail.append(m)
        print(f"  {nt:>8}      " + "".join(f"{s:>+12.2f}" for s in row))

    print(f"\n{'-'*94}")
    print("detail (occ = mean train queries per populated bucket; "
          "stdComp -> 1 means variation preserved)")
    print(f"  {'n_train':>8} {'bits':>5} {'RMSE':>7} {'skill':>7} {'occ':>7} "
          f"{'stdComp':>8} {'biasLo':>8} {'biasHi':>8} {'hit%':>6}")
    for m in detail:
        print(f"  {m['n_train']:>8} {m['bits']:>5} {m['rmse']:>7.1f} "
              f"{m['skill']:>+7.2f} {m['mean_occupancy']:>7.1f} "
              f"{m['std_comp']:>8.2f} {m['bias_lo']:>+8.1f} {m['bias_hi']:>+8.1f} "
              f"{100*m['hit_frac']:>5.0f}%")
    print(f"{'='*94}\n")

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results_sift1m" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "joint_bits_train.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_dir / 'joint_bits_train.json'}")


if __name__ == "__main__":
    main()
