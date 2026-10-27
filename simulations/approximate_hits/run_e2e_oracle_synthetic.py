#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Best-case approximate-hit experiment on SIFT with a synthetic skewed workload.

Workload follows Aker's simZipf (Oh et al., PVLDB 19(10), Sec. 5.1):
  1. Similar-query groups: for each base query q (official SIFT query set),
     build `group_size` variants.
       interp (Aker): q' = (1 - lam) q + lam p, p another base query,
                      lam ~ U(0, lambda_max) (Aker uses lambda_max = 0.5)
       gauss:         q' = q + noise, ||noise|| ~ noise_rel * ||q||
  2. Request sequence: groups get Zipf(s) popularity over a random rank
     order; each request picks a group by popularity, then a variant
     uniformly.

Candidate-generating, scoring, and test request sequences are independent
draws from the same workload. Repeated vectors are deduplicated and weighted.
The database, retention oracle, policy, and greedy population are the same as
run_e2e_oracle_sift.py.
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
from simulations.approximate_hits import run_e2e_oracle_sift as base
from simulations.approximate_hits.greedy_population import GreedyPopulator
from simulations.approximate_hits.run_e2e_oracle_sift import (
    DEFAULT_CHECKPOINTS, DEFAULT_K_VALUES, DEFAULT_TARGETS, SIFT1M_DIR,
    exact_topk, plot_results, run_target, _print_target,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------


class SimZipfWorkload:
    """Lazily materialized similar-query groups with Zipf group popularity."""

    def __init__(self, base_queries, group_size, zipf_s, method,
                 lambda_max, noise_rel, seed):
        self.q = base_queries.astype(np.float64)
        self.n_groups = len(base_queries)
        self.group_size = group_size
        self.method = method
        self.noise_rel = noise_rel
        self.seed = seed
        rng = np.random.default_rng(seed)
        # interpolation partners (never the anchor itself) and mixing weights
        partners = rng.integers(0, self.n_groups - 1, size=(self.n_groups, group_size))
        partners += partners >= np.arange(self.n_groups)[:, None]
        self.partners = partners
        self.lam = rng.uniform(0.0, lambda_max, size=(self.n_groups, group_size))
        # Zipf popularity over a random rank order of groups
        rank_of_group = rng.permutation(self.n_groups)
        p = 1.0 / np.power(rank_of_group + 1.0, zipf_s)
        self.group_prob = p / p.sum()

    def sample(self, n, rng):
        """Sample n requests; returns variant ids (group * group_size + j)."""
        g = rng.choice(self.n_groups, size=n, p=self.group_prob)
        j = rng.integers(0, self.group_size, size=n)
        return g * self.group_size + j

    def vectors(self, ids):
        g = ids // self.group_size
        j = ids % self.group_size
        anchor = self.q[g]
        if self.method == "interp":
            lam = self.lam[g, j][:, None]
            return (1.0 - lam) * anchor + lam * self.q[self.partners[g, j]]
        out = np.empty_like(anchor)
        dim = anchor.shape[1]
        for i, vid in enumerate(ids.tolist()):
            z = np.random.default_rng([self.seed, vid]).standard_normal(dim)
            out[i] = anchor[i] + self.noise_rel * np.linalg.norm(anchor[i]) * z / np.sqrt(dim)
        return out


def _dedup(ids):
    uniq, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return uniq, counts.astype(np.int64), inv


def build_inputs(args):
    """Database and workload draws; deterministic given args (sizes, seed).

    Returns a dict with db, the workload, deduplicated candidate / scoring /
    test variant ids, their weights, the test access order, and stats.
    """
    # same database subsample as run_e2e_oracle_sift.py for the same seed
    rng = np.random.default_rng(args.seed)
    logger.info("Loading SIFT1M...")
    sift_base = load_fvecs_fast(str(SIFT1M_DIR / "sift_base.fvecs"))
    db = sift_base[rng.choice(len(sift_base), size=args.n_db, replace=False)]
    base_queries = load_fvecs_fast(str(SIFT1M_DIR / "sift_query.fvecs"))

    wl = SimZipfWorkload(base_queries, args.group_size, args.zipf, args.method,
                         args.lambda_max, args.noise_rel, args.seed)
    wrng = np.random.default_rng(args.seed + 1)
    cand_ids, _, _ = _dedup(wl.sample(args.n_cand_requests, wrng))
    score_ids, score_w, _ = _dedup(wl.sample(args.n_score_requests, wrng))
    test_req = wl.sample(args.n_test_requests, wrng)
    test_ids, test_w, test_seq = _dedup(test_req)

    stats = {
        "distinct_cand": int(len(cand_ids)),
        "distinct_score": int(len(score_ids)),
        "distinct_test": int(len(test_ids)),
        "test_requests_exact_repeat_of_score": float(np.isin(test_req, score_ids).mean()),
        "test_requests_group_in_cand": float(np.isin(
            test_req // args.group_size, cand_ids // args.group_size).mean()),
        "top1pct_groups_request_share": float(
            np.sort(wl.group_prob)[::-1][:max(1, wl.n_groups // 100)].sum()),
    }
    return {"db": db, "wl": wl, "cand_ids": cand_ids, "score_ids": score_ids,
            "score_w": score_w, "test_ids": test_ids, "test_w": test_w,
            "test_seq": test_seq, "stats": stats}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--zipf", type=float, default=0.99, help="Zipf exponent s")
    ap.add_argument("--method", choices=["interp", "gauss"], default="interp")
    ap.add_argument("--lambda-max", type=float, default=0.5)
    ap.add_argument("--noise-rel", type=float, default=0.1)
    ap.add_argument("--group-size", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-db", type=int, default=50000)
    ap.add_argument("--n-cand-requests", type=int, default=5000)
    ap.add_argument("--n-score-requests", type=int, default=20000)
    ap.add_argument("--n-test-requests", type=int, default=100000)
    ap.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    ap.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--checkpoints", type=float, nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--max-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    inp = build_inputs(args)
    db, wl, stats = inp["db"], inp["wl"], inp["stats"]
    cand_ids, score_ids, score_w = inp["cand_ids"], inp["score_ids"], inp["score_w"]
    test_ids, test_w, test_seq = inp["test_ids"], inp["test_w"], inp["test_seq"]
    logger.info(f"Workload stats: {stats}")

    k_max = max(args.k_values)
    logger.info(f"Exact top-{k_max} for {len(cand_ids)} candidate queries...")
    cand_topk = exact_topk(wl.vectors(cand_ids), db, k_max)
    logger.info(f"Exact top-{args.k} for scoring and test queries...")
    score_topk = exact_topk(wl.vectors(score_ids), db, args.k)
    test_topk = exact_topk(wl.vectors(test_ids), db, args.k)

    cand_vectors = [cand_topk[i, :K] for i in range(len(cand_ids)) for K in args.k_values]
    logger.info(f"{len(cand_vectors)} candidates")

    max_size = int(args.max_frac * args.n_db)
    checkpoints_abs = sorted(int(c * args.n_db) for c in args.checkpoints
                             if c <= args.max_frac)

    shared = dict(cand_vectors=cand_vectors, score_topk=score_topk,
                  test_topk=test_topk, n_db=args.n_db, k=args.k,
                  max_size=max_size, checkpoints_abs=checkpoints_abs,
                  score_weights=score_w, test_weights=test_w,
                  test_sequence=test_seq)
    if args.workers > 1:
        import multiprocessing as mp
        logger.info("Building shared greedy index...")
        shared["template"] = GreedyPopulator(
            cand_vectors, score_topk, args.n_db, args.k, 1.0, eval_topk=test_topk,
            score_weights=score_w, eval_weights=test_w)
        logger.info(f"  pairs={shared['template'].n_pairs} "
                    f"triples={shared['template'].n_triples}")
        base._SHARED.update(shared)
        with mp.get_context("fork").Pool(min(args.workers, len(args.targets))) as pool:
            results = pool.map(base._run_target_worker, args.targets)
    else:
        results = [run_target(t, **shared) for t in args.targets]
    for res in results:
        _print_target(res)

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": vars(args), "workload_stats": stats,
                   "results": results}, f)
    plot_results(results, out_dir, args.n_db,
                 title=f"SIFT simZipf s={args.zipf} ({args.method})")
    logger.info(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
