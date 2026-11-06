#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Reproduce the Stage 1 vs Stage 3 radius-estimation asymmetry on real SIFT data.

Stages (following Anupta):
  Stage 1: uniform database,        queries sampled from SIFT.
  Stage 3: SIFT database subsample, queries sampled from SIFT.

We expect:
  Stage 1 -> symmetric regression to the mean (over-est low, under-est high).
  Stage 3 -> asymmetric: low radii ~accurate, high radii under-estimated.

Diagnostics recorded per test query to discriminate explanations:
  - n_train_avg, hamming_used : tests the empty-bucket / fallback story (C).
  - binned bias vs true radius : shows the shape of the asymmetry.
  - per-bin mean train-count    : are high-radius buckets starved of train data?
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from datasets.dataloaders import load_fvecs
from simulations.radius_estimation.lsh_radius_estimator import (
    LSHRadiusEstimator, true_radius,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

SIFT_DIR = Path("datasets/sift/raw_siftsmall/siftsmall")

#-------------------------------------------------------------------------------


def _binned(r_true, values, n_bins=10):
    """Mean of `values` within equal-count bins of r_true. Returns centers, means."""
    mask = np.isfinite(values) & np.isfinite(r_true)
    r = r_true[mask]
    v = values[mask]
    order = np.argsort(r)
    r, v = r[order], v[order]
    edges = np.linspace(0, len(r), n_bins + 1).astype(int)
    centers, means = [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        centers.append(float(np.mean(r[lo:hi])))
        means.append(float(np.mean(v[lo:hi])))
    return centers, means


def _uniform_like(data, n, rng):
    """Uniform database over the per-dimension range of `data`."""
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    return rng.uniform(lo, hi, size=(n, data.shape[1])).astype(np.float32)


def run_stage(name, database, train_q, test_q, k, n_bits, hamming, seed):
    r_train = true_radius(train_q, database, k)
    r_test = true_radius(test_q, database, k)
    gm = float(np.mean(r_train))

    est = LSHRadiusEstimator(n_bits=n_bits, dim=database.shape[1], seed=seed)
    est.fit(train_q, r_train)
    out = est.estimate(test_q, hamming=hamming, expand_fallback=True, global_mean=gm)
    estimate = out["estimate"]
    error = estimate - r_test

    m = np.isfinite(estimate)
    std_comp = float(np.std(estimate[m]) / np.std(r_test[m]))
    centers, bias = _binned(r_test, error)
    _, bin_count = _binned(r_test, out["n_train_avg"].astype(float))
    _, bin_ham = _binned(r_test, out["hamming_used"].astype(float))

    summary = {
        "stage": name,
        "n_database": int(len(database)),
        "n_train": int(len(train_q)),
        "n_test": int(len(test_q)),
        "k": k, "n_bits": n_bits, "hamming": hamming, "dim": int(database.shape[1]),
        "true_radius_mean": float(np.mean(r_test)),
        "true_radius_skew": _skew(r_test),
        "mean_error": float(np.nanmean(error)),
        "std_compression": std_comp,
        "fallback_frac": float(np.mean(out["hamming_used"] > n_bits)),
        "bin_centers": centers,
        "bin_bias": bias,
        "bin_train_count": bin_count,
        "bin_hamming_used": bin_ham,
    }
    return summary


def _skew(x):
    x = x[np.isfinite(x)]
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _print_stage(s):
    print(f"\n{'='*78}")
    print(f"{s['stage']}  (db={s['n_database']}, train={s['n_train']}, "
          f"test={s['n_test']}, k={s['k']}, bits={s['n_bits']}, H={s['hamming']})")
    print(f"{'='*78}")
    print(f"  true radius mean={s['true_radius_mean']:.1f}  skew={s['true_radius_skew']:+.2f}")
    print(f"  std compression={s['std_compression']:.2f}  "
          f"mean error={s['mean_error']:+.2f}  fallback={s['fallback_frac']:.1%}")
    print(f"  {'radius bin':>12} {'bias':>10} {'train/bkt':>10} {'H used':>8}")
    for c, b, n, h in zip(s['bin_centers'], s['bin_bias'],
                          s['bin_train_count'], s['bin_hamming_used']):
        print(f"  {c:>12.1f} {b:>+10.2f} {n:>10.0f} {h:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-bits", type=int, default=12)
    ap.add_argument("--hamming", type=int, default=0)
    ap.add_argument("--n-db", type=int, default=10000)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    logger.info("Loading SIFT vectors...")
    base = load_fvecs(str(SIFT_DIR / "siftsmall_base.fvecs"))      # 10k x 128
    learn = load_fvecs(str(SIFT_DIR / "siftsmall_learn.fvecs"))    # 25k x 128
    logger.info(f"  base={base.shape}  learn={learn.shape}")

    # SIFT database subsample (real, clumpy structure)
    db_idx = rng.choice(len(base), size=min(args.n_db, len(base)), replace=False)
    sift_db = base[db_idx]

    # SIFT-sampled queries: drawn from the learn set (distinct from base)
    q_total = args.n_train + args.n_test
    q_idx = rng.choice(len(learn), size=min(q_total, len(learn)), replace=False)
    sift_queries = learn[q_idx]
    train_q = sift_queries[:args.n_train]
    test_q = sift_queries[args.n_train:args.n_train + args.n_test]

    # uniform database matched to SIFT coordinate ranges
    uni_db = _uniform_like(base, len(sift_db), rng)

    # Stage 1: uniform DB, SIFT queries
    s1 = run_stage("Stage 1: uniform DB, SIFT queries",
                   uni_db, train_q, test_q, args.k, args.n_bits, args.hamming, args.seed)
    # Stage 3: SIFT DB, SIFT queries
    s3 = run_stage("Stage 3: SIFT DB, SIFT queries",
                   sift_db, train_q, test_q, args.k, args.n_bits, args.hamming, args.seed)

    _print_stage(s1)
    _print_stage(s3)

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results_sift" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stages.json", "w") as f:
        json.dump([s1, s3], f, indent=2)
    logger.info(f"Wrote {out_dir / 'stages.json'}")


if __name__ == "__main__":
    main()
