#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Train-query distribution control on real SIFT (50k subsample).

Hypothesis under test: the Stage-3 asymmetry (high-radius under-estimation
without matching low-radius over-estimation) is caused by sparse-region
starvation, which requires TRAIN queries to follow the data distribution (so
sparse-region buckets are nearly empty and borrowing reaches into dense,
small-radius clumps).

Prediction:
  - train queries DATA-FOLLOWING  -> asymmetry present, train-count collapses
    at high radii, Hamming borrowing escalates at high radii.
  - train queries UNIFORM-IN-SPACE -> asymmetry suppressed, train-count roughly
    flat across radius bins, borrowing does not track radius.

Test queries are always SIFT-sampled (that is the workload we care about);
only the TRAIN distribution changes. Database is always the SIFT subsample.
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

from simulations.radius_estimation.lsh_radius_estimator import (
    LSHRadiusEstimator, true_radius,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

SIFT1M_DIR = Path("datasets/sift/raw_sift1m/sift")

#-------------------------------------------------------------------------------


def _binned(r_true, values, n_bins=10):
    mask = np.isfinite(values) & np.isfinite(r_true)
    r, v = r_true[mask], values[mask]
    order = np.argsort(r)
    r, v = r[order], v[order]
    edges = np.linspace(0, len(r), n_bins + 1).astype(int)
    centers, means = [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            centers.append(float(np.mean(r[lo:hi])))
            means.append(float(np.mean(v[lo:hi])))
    return centers, means


def _skew(x):
    x = x[np.isfinite(x)]
    s = np.std(x)
    return 0.0 if s == 0 else float(np.mean(((x - np.mean(x)) / s) ** 3))


def _uniform_in_space(ref, n, rng):
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    return rng.uniform(lo, hi, size=(n, ref.shape[1])).astype(np.float32)


def run_variant(name, database, train_q, test_q, r_test, k, n_bits, hamming, seed):
    r_train = true_radius(train_q, database, k)
    gm = float(np.mean(r_train))
    est = LSHRadiusEstimator(n_bits=n_bits, dim=database.shape[1], seed=seed)
    est.fit(train_q, r_train)
    out = est.estimate(test_q, hamming=hamming, expand_fallback=True, global_mean=gm)
    estimate = out["estimate"]
    error = estimate - r_test
    m = np.isfinite(estimate)

    centers, bias = _binned(r_test, error)
    _, bin_count = _binned(r_test, out["n_train_avg"].astype(float))
    _, bin_ham = _binned(r_test, out["hamming_used"].astype(float))
    return {
        "variant": name,
        "mean_error": float(np.nanmean(error)),
        "std_compression": float(np.std(estimate[m]) / np.std(r_test[m])),
        "fallback_frac": float(np.mean(out["hamming_used"] > n_bits)),
        "bin_centers": centers, "bin_bias": bias,
        "bin_train_count": bin_count, "bin_hamming_used": bin_ham,
    }


def _print(v, skew=None):
    print(f"\n{'='*82}")
    print(f"{v['variant']}")
    print(f"{'='*82}")
    if skew is not None:
        print(f"  true radius skew={skew:+.2f}")
    print(f"  std compression={v['std_compression']:.2f}  "
          f"mean error={v['mean_error']:+.1f}  fallback={v['fallback_frac']:.1%}")
    print(f"  {'radius bin':>12} {'bias':>10} {'train/bkt':>10} {'H used':>8}")
    for c, b, n, h in zip(v['bin_centers'], v['bin_bias'],
                          v['bin_train_count'], v['bin_hamming_used']):
        print(f"  {c:>12.1f} {b:>+10.1f} {n:>10.0f} {h:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-bits", type=int, default=16)
    ap.add_argument("--hamming", type=int, default=0)
    ap.add_argument("--n-db", type=int, default=50000)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    logger.info("Loading SIFT1M base subsample + learn queries...")
    # read enough base rows to subsample n_db from, and learn for queries
    base = load_fvecs_fast(str(SIFT1M_DIR / "sift_base.fvecs"), max_n=max(args.n_db * 2, 100000))
    learn = load_fvecs_fast(str(SIFT1M_DIR / "sift_learn.fvecs"), max_n=args.n_train + args.n_test + 1000)
    logger.info(f"  base read={base.shape}  learn read={learn.shape}")

    db_idx = rng.choice(len(base), size=args.n_db, replace=False)
    sift_db = base[db_idx]

    # test queries: always SIFT-sampled (the real workload)
    q_idx = rng.choice(len(learn), size=args.n_test, replace=False)
    test_q = learn[q_idx]
    r_test = true_radius(test_q, sift_db, args.k)
    skew = _skew(r_test)
    logger.info(f"  test radius skew={skew:+.2f}")

    # train variant 1: data-following (sampled from SIFT learn, like test)
    remaining = np.setdiff1d(np.arange(len(learn)), q_idx)
    tr_idx = rng.choice(remaining, size=min(args.n_train, len(remaining)), replace=False)
    train_following = learn[tr_idx]

    # train variant 2: uniform in the data's coordinate box
    train_uniform = _uniform_in_space(sift_db, args.n_train, rng)

    v_follow = run_variant(
        "TRAIN = data-following (SIFT-sampled)",
        sift_db, train_following, test_q, r_test,
        args.k, args.n_bits, args.hamming, args.seed,
    )
    v_uniform = run_variant(
        "TRAIN = uniform-in-space",
        sift_db, train_uniform, test_q, r_test,
        args.k, args.n_bits, args.hamming, args.seed,
    )

    _print(v_follow, skew)
    _print(v_uniform, skew)

    print(f"\n{'='*82}")
    print("CONTROL READOUT (does the asymmetry depend on train distribution?)")
    print(f"{'='*82}")
    for v in (v_follow, v_uniform):
        lo_cnt = v['bin_train_count'][0]
        hi_cnt = v['bin_train_count'][-1]
        lo_bias = v['bin_bias'][0]
        hi_bias = v['bin_bias'][-1]
        print(f"  {v['variant']:<40}")
        print(f"      train/bkt  low-radius={lo_cnt:.0f}  high-radius={hi_cnt:.0f}  "
              f"(ratio {lo_cnt / max(hi_cnt, 1e-9):.1f}x)")
        print(f"      bias       low-radius={lo_bias:+.1f}  high-radius={hi_bias:+.1f}")
    print(f"{'='*82}\n")

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results_sift1m" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_control.json", "w") as f:
        json.dump({"test_radius_skew": skew,
                   "data_following": v_follow, "uniform": v_uniform}, f, indent=2)
    logger.info(f"Wrote {out_dir / 'train_control.json'}")


if __name__ == "__main__":
    main()
