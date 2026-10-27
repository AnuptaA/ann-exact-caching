#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Granularity sweep for the LSH radius estimator on real SIFT (data-following).

Both train and test queries follow the data distribution (SIFT-sampled);
database is a SIFT subsample. We report, as LSH granularity grows:

  - RMSE and mean error (bias) of the estimate vs the true radius
  - the same for a GLOBAL-AVERAGE baseline (ignore the query, always predict
    the mean train radius)
  - skill = 1 - RMSE_lsh / RMSE_global  (how much better than global average)

Two granularity knobs:
  - bits per table (finer buckets, smaller neighborhoods)
  - number of tables L (pool train queries that collide in ANY table, then
    average their radii over the union; more tables -> more neighbors pooled)

Metrics:
  RMSE          = sqrt(mean((estimate - true)^2))
  mean_error    = mean(estimate - true)  (signed bias)
  bias_lo/hi    = mean error in lowest/highest true-radius decile
  std_comp      = std(estimate)/std(true) (1.0 = preserves variation)
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from itertools import combinations

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from datasets.dataloaders import load_fvecs_fast

from simulations.radius_estimation.lsh_radius_estimator import true_radius

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

SIFT1M_DIR = Path("datasets/sift/raw_sift1m/sift")

#-------------------------------------------------------------------------------


class MultiTableLSH:
    """L independent random-hyperplane tables for radius estimation.

    Estimate(q) = mean radius over the UNION of train queries that share q's
    bucket in at least one table, expanding Hamming per table if empty.
    """

    def __init__(self, n_tables, n_bits, dim, seed):
        self.L = n_tables
        self.n_bits = n_bits
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.planes = [rng.standard_normal((dim, n_bits)) for _ in range(n_tables)]
        self._pow2 = (1 << np.arange(n_bits)).astype(np.int64)
        self.tables = None  # list of dict: code -> set(train_idx)
        self.train_radii = None

    def _codes(self, vecs, t):
        bits = (vecs @ self.planes[t]) > 0.0
        return bits.astype(np.int64) @ self._pow2

    def fit(self, train_q, train_radii):
        self.train_radii = train_radii
        self.tables = []
        for t in range(self.L):
            codes = self._codes(train_q, t)
            tab = {}
            for i, c in enumerate(codes.tolist()):
                tab.setdefault(c, []).append(i)
            self.tables.append(tab)

    def _neighbors(self, code, max_h):
        yield code
        for h in range(1, max_h + 1):
            for flips in combinations(range(self.n_bits), h):
                m = code
                for b in flips:
                    m ^= (1 << b)
                yield m

    def estimate(self, test_q, hamming, global_mean):
        n = len(test_q)
        est = np.full(n, np.nan)
        n_pooled = np.zeros(n, dtype=np.int64)
        all_codes = [self._codes(test_q, t) for t in range(self.L)]
        for i in range(n):
            pooled = set()
            for t in range(self.L):
                code = int(all_codes[t][i])
                h = hamming
                found = set()
                while True:
                    for m in self._neighbors(code, h):
                        lst = self.tables[t].get(m)
                        if lst:
                            found.update(lst)
                    if found or h >= self.n_bits:
                        break
                    h += 1
                pooled |= found
            if pooled:
                idx = np.fromiter(pooled, dtype=np.int64)
                est[i] = float(np.mean(self.train_radii[idx]))
                n_pooled[i] = len(idx)
            else:
                est[i] = global_mean
        return est, n_pooled


def _metrics(estimate, r_test, global_mean):
    m = np.isfinite(estimate)
    e = estimate[m] - r_test[m]
    rmse = float(np.sqrt(np.mean(e ** 2)))
    rmse_global = float(np.sqrt(np.mean((global_mean - r_test[m]) ** 2)))
    # deciles
    order = np.argsort(r_test[m])
    es = e[order]
    nlo = len(es) // 10
    bias_lo = float(np.mean(es[:nlo])) if nlo else float("nan")
    bias_hi = float(np.mean(es[-nlo:])) if nlo else float("nan")
    return {
        "rmse": rmse,
        "rmse_global": rmse_global,
        "skill": float(1.0 - rmse / rmse_global) if rmse_global > 0 else float("nan"),
        "mean_error": float(np.mean(e)),
        "std_comp": float(np.std(estimate[m]) / np.std(r_test[m])),
        "bias_lo": bias_lo,
        "bias_hi": bias_hi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hamming", type=int, default=0)
    ap.add_argument("--n-db", type=int, default=50000)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--bits-list", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--tables-list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--fixed-bits", type=int, default=16,
                    help="bits held fixed while sweeping tables")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    logger.info("Loading SIFT1M...")
    base = load_fvecs_fast(str(SIFT1M_DIR / "sift_base.fvecs"), max_n=max(args.n_db * 2, 100000))
    learn = load_fvecs_fast(str(SIFT1M_DIR / "sift_learn.fvecs"),
                       max_n=args.n_train + args.n_test + 2000)

    db_idx = rng.choice(len(base), size=args.n_db, replace=False)
    sift_db = base[db_idx]

    q_idx = rng.choice(len(learn), size=args.n_test + args.n_train, replace=False)
    test_q = learn[q_idx[:args.n_test]]
    train_q = learn[q_idx[args.n_test:args.n_test + args.n_train]]

    logger.info("Computing true radii (train + test)...")
    r_test = true_radius(test_q, sift_db, args.k)
    r_train = true_radius(train_q, sift_db, args.k)
    global_mean = float(np.mean(r_train))
    logger.info(f"  true radius: mean={np.mean(r_test):.1f} std={np.std(r_test):.1f}")
    logger.info(f"  global-average baseline RMSE = "
                f"{np.sqrt(np.mean((global_mean - r_test)**2)):.1f}")

    results = {"config": vars(args),
               "true_mean": float(np.mean(r_test)),
               "true_std": float(np.std(r_test)),
               "global_mean": global_mean,
               "bits_sweep": [], "tables_sweep": []}

    # --- sweep bits (single table) ---
    print(f"\n{'='*86}")
    print(f"BITS SWEEP (L=1 table, Hamming={args.hamming})")
    print(f"{'='*86}")
    print(f"  {'bits':>5} {'RMSE':>9} {'RMSEglob':>9} {'skill':>7} "
          f"{'bias':>8} {'stdComp':>8} {'biasLo':>8} {'biasHi':>8}")
    for bits in args.bits_list:
        lsh = MultiTableLSH(1, bits, sift_db.shape[1], args.seed)
        lsh.fit(train_q, r_train)
        est, _ = lsh.estimate(test_q, args.hamming, global_mean)
        m = _metrics(est, r_test, global_mean)
        m["bits"] = bits
        results["bits_sweep"].append(m)
        print(f"  {bits:>5} {m['rmse']:>9.1f} {m['rmse_global']:>9.1f} "
              f"{m['skill']:>+7.2f} {m['mean_error']:>+8.1f} {m['std_comp']:>8.2f} "
              f"{m['bias_lo']:>+8.1f} {m['bias_hi']:>+8.1f}")

    # --- sweep tables (fixed bits) ---
    print(f"\n{'='*86}")
    print(f"TABLES SWEEP (bits={args.fixed_bits}, Hamming={args.hamming})")
    print(f"{'='*86}")
    print(f"  {'L':>5} {'RMSE':>9} {'RMSEglob':>9} {'skill':>7} "
          f"{'bias':>8} {'stdComp':>8} {'biasLo':>8} {'biasHi':>8}")
    for L in args.tables_list:
        lsh = MultiTableLSH(L, args.fixed_bits, sift_db.shape[1], args.seed)
        lsh.fit(train_q, r_train)
        est, _ = lsh.estimate(test_q, args.hamming, global_mean)
        m = _metrics(est, r_test, global_mean)
        m["tables"] = L
        results["tables_sweep"].append(m)
        print(f"  {L:>5} {m['rmse']:>9.1f} {m['rmse_global']:>9.1f} "
              f"{m['skill']:>+7.2f} {m['mean_error']:>+8.1f} {m['std_comp']:>8.2f} "
              f"{m['bias_lo']:>+8.1f} {m['bias_hi']:>+8.1f}")
    print(f"{'='*86}\n")

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results_sift1m" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "granularity_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_dir / 'granularity_sweep.json'}")


if __name__ == "__main__":
    main()
