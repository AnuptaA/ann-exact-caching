#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Best-case end-to-end approximate-hit experiment on SIFT (oracle radius).

Setup:
  - Database: random subsample of SIFT1M base.
  - Train queries: SIFT learn set, split into candidate-generating queries
    (their top-K balls are cache candidates) and scoring queries (used to
    compute greedy utility). The split avoids a candidate covering its own
    source query.
  - Test queries: official SIFT query set (held out).
  - Retention uses each query's TRUE top-k (exact backend, oracle radius).
  - Policy: hits-only running average >= target (best-case offline admission;
    the online order-dependent variant is also reported at checkpoints).

Output: hit rate (fraction of test queries served from cache) vs cache size
(fraction of database vectors), one line per retention target.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.dataloaders import load_fvecs_fast
from simulations.approximate_hits.greedy_population import (
    GreedyPopulator,
    hits_from_histograms,
    online_hits,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SIFT1M_DIR = Path("datasets/sift/raw_sift1m/sift")
DEFAULT_TARGETS = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
DEFAULT_K_VALUES = [10, 20, 50, 100, 200, 500]
DEFAULT_CHECKPOINTS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

#-------------------------------------------------------------------------------


def exact_topk(queries: np.ndarray, db: np.ndarray, k: int, chunk: int = 1000) -> np.ndarray:
    """Exact top-k ids (sorted by distance) under squared Euclidean distance.

    Large databases (> 2M vectors) use exact_topk_blocked to bound memory.
    """
    if len(db) > 2_000_000:
        return exact_topk_blocked(queries, db, k)
    db64 = db.astype(np.float64)
    db_sq = np.einsum("ij,ij->i", db64, db64)
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        q = queries[s:s + chunk].astype(np.float64)
        d = db_sq[None, :] - 2.0 * (q @ db64.T)
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, axis=1), axis=1)
        out[s:s + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def exact_topk_blocked(queries: np.ndarray, db: np.ndarray, k: int,
                       q_chunk: int = 4096, db_block: int = 500_000) -> np.ndarray:
    """Exact top-k ids for large databases, blocked over queries and database.

    Two stages per database block. Screen: one float32 matmul of [q, 1] with
    [x, -|x|^2/2] gives s = q.x - |x|^2/2 (larger s = closer, since
    |q - x|^2 = |q|^2 - 2s); keep vectors with s >= (threshold implied by the
    current k-th best) - slack. Refine: exact float64 distances for the
    survivors, merged into the running top-k. Every true top-k vector passes
    the screen, so the result is exact. Ties are broken by lower id. Output
    rows are sorted by (distance, id).
    """
    n, dim = db.shape
    xa = np.empty((n, dim + 1), dtype=np.float32)
    db_sq = np.empty(n, dtype=np.float64)
    for b in range(0, n, db_block):
        xb = np.asarray(db[b:b + db_block], dtype=np.float64)
        db_sq[b:b + len(xb)] = np.einsum("ij,ij->i", xb, xb)
        xa[b:b + len(xb), :dim] = xb
        xa[b:b + len(xb), dim] = -0.5 * db_sq[b:b + len(xb)]
    max_db_sq = float(db_sq.max())
    out = np.empty((len(queries), k), dtype=np.int64)
    buf = np.empty((min(q_chunk, len(queries)), min(db_block, n)), dtype=np.float32)
    for s0 in range(0, len(queries), q_chunk):
        q64 = np.asarray(queries[s0:s0 + q_chunk], dtype=np.float64)
        c = len(q64)
        qa = np.ones((c, dim + 1), dtype=np.float32)
        qa[:, :dim] = q64
        q_sq = np.einsum("ij,ij->i", q64, q64)
        # float32 screening error is far below this slack
        slack = 1e-4 * (float(q_sq.max()) + max_db_sq + 1.0)
        best_d = np.full((c, k), np.inf)
        best_i = np.full((c, k), -1, dtype=np.int64)
        for b in range(0, n, db_block):
            nb = min(db_block, n - b)
            s = np.matmul(qa, xa[b:b + nb].T, out=buf[:c, :nb])
            if np.isinf(best_d[:, -1]).any():
                thr = np.partition(s, nb - k, axis=1)[:, nb - k].astype(np.float64)
                known = ~np.isinf(best_d[:, -1])
                thr[known] = np.maximum(thr[known], 0.5 * (q_sq[known] - best_d[known, -1]))
            else:
                thr = 0.5 * (q_sq - best_d[:, -1])
            rows, cols = np.nonzero(s >= (thr - slack)[:, None].astype(np.float32))
            ids = cols.astype(np.int64) + b
            diff = q64[rows] - np.asarray(db[ids], dtype=np.float64)
            d_new = np.einsum("ij,ij->i", diff, diff)
            # merge: k smallest by (distance, id) per row
            all_r = np.concatenate([np.repeat(np.arange(c), k), rows])
            all_d = np.concatenate([best_d.ravel(), d_new])
            all_i = np.concatenate([best_i.ravel(), ids])
            order = np.lexsort((all_i, all_d, all_r))
            all_r, all_d, all_i = all_r[order], all_d[order], all_i[order]
            starts = np.searchsorted(all_r, np.arange(c))
            keep = (np.arange(len(all_r)) - starts[all_r]) < k
            best_d = all_d[keep].reshape(c, k)
            best_i = all_i[keep].reshape(c, k)
        out[s0:s0 + c] = best_i
    return out


def run_target(target, cand_vectors, score_topk, test_topk, n_db, k,
               max_size, checkpoints_abs, score_weights=None, test_weights=None,
               test_sequence=None, template=None):
    """Greedy population for one target; evaluates on test queries.

    score_weights / test_weights: multiplicities of distinct queries.
    test_sequence: test access order as indices into test_topk (for the online
    policy); defaults to each distinct test query once, in order.
    template: optional unstepped GreedyPopulator built from the same inputs;
    cloned instead of rebuilding (shares large arrays across forked workers).
    """
    if template is not None:
        pop = template.clone_for_target(target)
    else:
        pop = GreedyPopulator(cand_vectors, score_topk, n_db, k, target,
                              eval_topk=test_topk, score_weights=score_weights,
                              eval_weights=test_weights)
    n_test = int(pop.eval.weights.sum())
    n_score = int(pop.score.weights.sum())

    def snapshot(counts):
        hist = pop.eval.histogram()
        h, kept = hits_from_histograms(hist, target, k)
        seq_counts = counts if test_sequence is None else counts[test_sequence]
        h_on, kept_on = online_hits(seq_counts, target, k)
        n_seq = len(seq_counts)
        hit_rate = int(h) / n_test
        retained = int(kept) / (k * n_test)
        hit_rate_on = h_on / n_seq
        retained_on = kept_on / (k * n_seq)
        return {
            "cache_size": pop.cache_size,
            "cache_frac": pop.cache_size / n_db,
            "test_hit_rate": hit_rate,
            "test_retained_rate": retained,
            # all queries, misses counted as retention 1
            "test_effective_retention": 1.0 - hit_rate + retained,
            "test_hit_rate_online": hit_rate_on,
            "test_retained_rate_online": retained_on,
            "test_effective_retention_online": 1.0 - hit_rate_on + retained_on,
            "train_hit_rate": pop.score_hits / n_score,
            "n_entries": len(pop.selected),
        }

    curve = []  # every step: size, best-case test hits, train hits
    points = []
    next_cp = 0
    prev_counts = pop.eval.counts.copy()
    prev_snapshot = snapshot(prev_counts)
    while True:
        rec = pop.step(max_size)
        if rec is None:
            break
        # checkpoints crossed by this step: record the state just before it
        while next_cp < len(checkpoints_abs) and pop.cache_size > checkpoints_abs[next_cp]:
            p = dict(prev_snapshot)
            p["checkpoint"] = checkpoints_abs[next_cp] / n_db
            points.append(p)
            next_cp += 1
        prev_counts = pop.eval.counts.copy()
        prev_snapshot = snapshot(prev_counts)
        curve.append([pop.cache_size, prev_snapshot["test_hit_rate"],
                      prev_snapshot["train_hit_rate"]])
    final = dict(prev_snapshot)
    final["checkpoint"] = None
    points.append(final)  # final state (budget reached or greedy stopped)
    return {"target": target, "points": points, "curve": curve,
            "selected": [int(j) for j in pop.selected],
            "n_pairs": pop.n_pairs, "n_triples": pop.n_triples}


# shared inputs for forked workers (avoids pickling large arrays per task)
_SHARED: dict = {}


def _run_target_worker(target):
    return run_target(target, **_SHARED)


def plot_results(results, out_dir, n_db, title="SIFT"):
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    panels = [
        ("test_hit_rate", "hit rate (best-case admission)"),
        ("test_hit_rate_online", "hit rate (online admission)"),
    ]
    for scale in ["linear", "loglog"]:
        log = scale == "loglog"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
        for ax, (key, panel_title) in zip(axes, panels):
            for res, color in zip(results, colors):
                xs = np.array([p["cache_frac"] for p in res["points"]])
                ys = np.array([p[key] for p in res["points"]])
                keep = (xs > 0) & (ys > 0) if log else np.ones(len(xs), dtype=bool)
                order = np.argsort(xs[keep])
                ax.plot(xs[keep][order], ys[keep][order], "o-",
                        color=color, label=f"T={res['target']}")
            ref = np.logspace(-4, 0, 50) if log else np.array([0.0, 1.0])
            ax.plot(ref, ref, "k--", lw=1, label="y = x")
            x_max = max(p["cache_frac"] for r in results for p in r["points"]) * 1.1
            if log:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(5e-4, x_max)
                ax.set_ylim(1e-4, 1.05)
            else:
                ax.set_xlim(0.0, x_max)
                ax.set_ylim(0, 1)
            ax.set_xlabel("cache size (fraction of database vectors)")
            ax.set_ylabel("fraction of test queries")
            ax.set_title(panel_title, fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle(f"{title}, oracle radius, greedy population (hits-only average policy)")
        plt.tight_layout()
        fig.savefig(out_dir / f"hit_rate_vs_cache_size_{scale}.png", dpi=120)
        plt.close(fig)


def load_inputs(args):
    """Database, candidate-generating, scoring, and test query vectors.

    Deterministic given args (dataset, sizes, seed), so evaluation scripts can
    rebuild a run's exact inputs from its saved config.
    """
    rng = np.random.default_rng(args.seed)
    if args.dataset == "sift":
        logger.info("Loading SIFT1M...")
        base = load_fvecs_fast(str(SIFT1M_DIR / "sift_base.fvecs"))
        learn = load_fvecs_fast(str(SIFT1M_DIR / "sift_learn.fvecs"))
        test = load_fvecs_fast(str(SIFT1M_DIR / "sift_query.fvecs"))[:args.n_test]
        db = base[rng.choice(len(base), size=args.n_db, replace=False)]
        train_idx = rng.choice(len(learn), size=args.n_cand_queries + args.n_score_queries,
                               replace=False)
        cand_q = learn[train_idx[:args.n_cand_queries]]
        score_q = learn[train_idx[args.n_cand_queries:]]
    else:
        # ESCI: database = pre-built product subsample; train and test are
        # disjoint random splits of the real ESCI queries
        esci_dir = Path(args.esci_dir)
        logger.info(f"Loading ESCI from {esci_dir}...")
        db = np.load(esci_dir / f"esci_db{args.n_db}_s{args.seed}.npy")
        pool = np.load(esci_dir / "esci_queries.npy")
        perm = rng.permutation(len(pool))
        n_c, n_s = args.n_cand_queries, args.n_score_queries
        test = pool[perm[:args.n_test]]
        cand_q = pool[perm[args.n_test:args.n_test + n_c]]
        score_q = pool[perm[args.n_test + n_c:args.n_test + n_c + n_s]]
    return db, cand_q, score_q, test


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", choices=["sift", "esci"], default="sift")
    ap.add_argument("--esci-dir", type=str, default="datasets/esci/data_vectors")
    ap.add_argument("--k", type=int, default=10, help="test query result count")
    ap.add_argument("--n-db", type=int, default=50000)
    ap.add_argument("--n-cand-queries", type=int, default=5000)
    ap.add_argument("--n-score-queries", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
                    help="candidate ball sizes K")
    ap.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--checkpoints", type=float, nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--max-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes over targets")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    db, cand_q, score_q, test = load_inputs(args)
    logger.info(f"  db={db.shape} cand_queries={cand_q.shape} "
                f"score_queries={score_q.shape} test={test.shape}")

    k_max = max(args.k_values)
    logger.info(f"Exact top-{k_max} for candidate queries...")
    cand_topk = exact_topk(cand_q, db, k_max)
    logger.info(f"Exact top-{args.k} for scoring and test queries...")
    score_topk = exact_topk(score_q, db, args.k)
    test_topk = exact_topk(test, db, args.k)

    cand_vectors = [cand_topk[i, :K] for i in range(len(cand_q)) for K in args.k_values]
    logger.info(f"{len(cand_vectors)} candidates "
                f"({len(cand_q)} queries x K in {args.k_values})")

    max_size = int(args.max_frac * args.n_db)
    checkpoints_abs = sorted(int(c * args.n_db) for c in args.checkpoints
                             if c <= args.max_frac)

    results = []
    if args.workers > 1:
        import multiprocessing as mp
        _SHARED.update(cand_vectors=cand_vectors, score_topk=score_topk,
                       test_topk=test_topk, n_db=args.n_db, k=args.k,
                       max_size=max_size, checkpoints_abs=checkpoints_abs)
        logger.info("Building shared greedy index...")
        _SHARED["template"] = GreedyPopulator(cand_vectors, score_topk, args.n_db,
                                              args.k, 1.0, eval_topk=test_topk)
        logger.info(f"  pairs={_SHARED['template'].n_pairs} "
                    f"triples={_SHARED['template'].n_triples}")
        logger.info(f"Greedy population for targets {args.targets} "
                    f"({args.workers} workers)...")
        with mp.get_context("fork").Pool(min(args.workers, len(args.targets))) as pool:
            results = pool.map(_run_target_worker, args.targets)
        for res in results:
            _print_target(res)
    else:
        for target in args.targets:
            logger.info(f"Greedy population for target T={target}...")
            res = run_target(target, cand_vectors, score_topk, test_topk,
                             args.n_db, args.k, max_size, checkpoints_abs)
            results.append(res)
            logger.info(f"  pairs={res['n_pairs']} triples={res['n_triples']}")
            _print_target(res)

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": vars(args), "results": results}, f)
    plot_results(results, out_dir, args.n_db, title=args.dataset.upper())
    logger.info(f"Wrote {out_dir}")


def _print_target(res):
    print(f"\n  T={res['target']}")
    print(f"  {'cache%':>8} {'hit%':>7} {'eff%':>7} {'onl hit%':>9} {'onl eff%':>9} "
          f"{'train%':>7} {'entries':>8}")
    for p in res["points"]:
        print(f"  {100*p['cache_frac']:>8.2f} {100*p['test_hit_rate']:>7.2f} "
              f"{100*p['test_effective_retention']:>7.2f} "
              f"{100*p['test_hit_rate_online']:>9.2f} "
              f"{100*p['test_effective_retention_online']:>9.2f} "
              f"{100*p['train_hit_rate']:>7.2f} {p['n_entries']:>8}")


if __name__ == "__main__":
    main()
