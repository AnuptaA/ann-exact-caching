#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Two follow-up questions on real SIFT (data-following train + test, 50k db):

  (A) Does MORE TRAINING DATA help? Sweep n_train; measure RMSE, skill over the
      global-average baseline, and per-decile bias. Prediction: RMSE falls then
      plateaus (variance shrinks); per-decile smoothing bias persists.

  (B) Does a DENSITY-AWARE PARTITION help (put similar-density regions in the
      same bucket)? Compare three bucketings at matched bucket counts:
        1. plain random-hyperplane LSH          (baseline)
        2. local-density feature + 1D quantile bins
           (density estimated from TRAIN queries only -> realizable)
        3. ORACLE: bin directly on each point's true radius
           (uses test ground truth -> not realizable; an upper bound on how
            much any density-aware partition could help)

All estimators: estimate(q) = mean train radius in q's bucket, with a
global-mean fallback for empty buckets.
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


def _metrics(estimate, r_test, global_mean):
    m = np.isfinite(estimate)
    e = estimate[m] - r_test[m]
    rmse = float(np.sqrt(np.mean(e ** 2)))
    rmse_g = float(np.sqrt(np.mean((global_mean - r_test[m]) ** 2)))
    order = np.argsort(r_test[m])
    es = e[order]
    nlo = len(es) // 10
    return {
        "rmse": rmse,
        "skill": float(1 - rmse / rmse_g) if rmse_g > 0 else float("nan"),
        "mean_error": float(np.mean(e)),
        "std_comp": float(np.std(estimate[m]) / np.std(r_test[m])),
        "bias_lo": float(np.mean(es[:nlo])) if nlo else float("nan"),
        "bias_hi": float(np.mean(es[-nlo:])) if nlo else float("nan"),
    }


# ---- bucketing schemes: each returns (train_bucket_ids, test_bucket_ids) ----

def buckets_lsh(train_q, test_q, n_bits, dim, seed):
    rng = np.random.default_rng(seed)
    planes = rng.standard_normal((dim, n_bits))
    pow2 = (1 << np.arange(n_bits)).astype(np.int64)
    tb = ((train_q @ planes) > 0).astype(np.int64) @ pow2
    te = ((test_q @ planes) > 0).astype(np.int64) @ pow2
    return tb, te


def _knn_density_feature(q, train_ref, kd=10):
    """Local-density proxy: distance to the kd-th nearest TRAIN query.

    Estimated from train queries only (realizable at inference: a new point's
    density feature uses the train-query cloud, not the database ground truth).
    """
    n = len(q)
    feat = np.empty(n, dtype=np.float64)
    chunk = 512
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = np.linalg.norm(q[s:e, None, :] - train_ref[None, :, :], axis=2)
        feat[s:e] = np.partition(d, kd, axis=1)[:, kd]
    return feat


def buckets_density(train_q, test_q, n_buckets, seed, kd=10):
    # density feature for train and test from the same train reference cloud
    ftr = _knn_density_feature(train_q, train_q, kd)
    fte = _knn_density_feature(test_q, train_q, kd)
    edges = np.quantile(ftr, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    tb = np.digitize(ftr, edges[1:-1])
    te = np.digitize(fte, edges[1:-1])
    return tb, te


def buckets_oracle(r_train, r_test, n_buckets):
    """Bin directly on TRUE radius (uses test ground truth -> upper bound)."""
    edges = np.quantile(r_train, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    tb = np.digitize(r_train, edges[1:-1])
    te = np.digitize(r_test, edges[1:-1])
    return tb, te


def estimate_by_bucket(train_bucket, test_bucket, r_train, global_mean):
    sums, counts = {}, {}
    for b, r in zip(train_bucket.tolist(), r_train.tolist()):
        sums[b] = sums.get(b, 0.0) + r
        counts[b] = counts.get(b, 0) + 1
    est = np.empty(len(test_bucket))
    for i, b in enumerate(test_bucket.tolist()):
        est[i] = sums[b] / counts[b] if counts.get(b, 0) > 0 else global_mean
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-db", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-bits", type=int, default=16)
    ap.add_argument("--trainsizes", type=int, nargs="+",
                    default=[1000, 2000, 4000, 8000, 16000, 24000])
    ap.add_argument("--part-trainsize", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    logger.info("Loading SIFT1M...")
    base = load_fvecs_fast(str(SIFT1M_DIR / "sift_base.fvecs"), max_n=max(args.n_db * 2, 100000))
    maxtrain = max(max(args.trainsizes), args.part_trainsize)
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

    results = {"config": vars(args), "trainsweep": [], "partition": []}

    # ---- (A) train-size sweep, plain LSH ----
    print(f"\n{'='*80}")
    print(f"(A) TRAIN-SIZE SWEEP  (plain LSH, bits={args.n_bits})")
    print(f"{'='*80}")
    print(f"  {'n_train':>8} {'RMSE':>8} {'skill':>7} {'bias':>7} "
          f"{'stdComp':>8} {'biasLo':>8} {'biasHi':>8}")
    for nt in args.trainsizes:
        tr = train_pool[:nt]
        r_tr = r_train_pool[:nt]
        gm = float(np.mean(r_tr))
        tb, te = buckets_lsh(tr, test_q, args.n_bits, sift_db.shape[1], args.seed)
        est = estimate_by_bucket(tb, te, r_tr, gm)
        m = _metrics(est, r_test, gm)
        m["n_train"] = nt
        results["trainsweep"].append(m)
        print(f"  {nt:>8} {m['rmse']:>8.1f} {m['skill']:>+7.2f} "
              f"{m['mean_error']:>+7.1f} {m['std_comp']:>8.2f} "
              f"{m['bias_lo']:>+8.1f} {m['bias_hi']:>+8.1f}")

    # ---- (B) partition comparison at matched bucket counts ----
    nt = args.part_trainsize
    tr = train_pool[:nt]
    r_tr = r_train_pool[:nt]
    gm = float(np.mean(r_tr))
    # count populated LSH buckets to match the other schemes' resolution
    tb_lsh, te_lsh = buckets_lsh(tr, test_q, args.n_bits, sift_db.shape[1], args.seed)
    n_pop = len(set(tb_lsh.tolist()))
    logger.info(f"Partition comparison: n_train={nt}, matched buckets~{n_pop}")

    print(f"\n{'='*80}")
    print(f"(B) PARTITION COMPARISON  (n_train={nt}, ~{n_pop} buckets each)")
    print(f"{'='*80}")
    print(f"  {'scheme':<28} {'RMSE':>8} {'skill':>7} {'stdComp':>8} "
          f"{'biasLo':>8} {'biasHi':>8}")

    schemes = []
    schemes.append(("plain LSH", (tb_lsh, te_lsh)))
    schemes.append(("density-feature bins", buckets_density(tr, test_q, n_pop, args.seed, kd=args.k)))
    schemes.append(("ORACLE (true-radius bins)", buckets_oracle(r_tr, r_test, n_pop)))

    for name, (tb, te) in schemes:
        est = estimate_by_bucket(tb, te, r_tr, gm)
        m = _metrics(est, r_test, gm)
        m["scheme"] = name
        results["partition"].append(m)
        print(f"  {name:<28} {m['rmse']:>8.1f} {m['skill']:>+7.2f} "
              f"{m['std_comp']:>8.2f} {m['bias_lo']:>+8.1f} {m['bias_hi']:>+8.1f}")
    print(f"{'='*80}\n")

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results_sift1m" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trainsize_partition.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_dir / 'trainsize_partition.json'}")


if __name__ == "__main__":
    main()
