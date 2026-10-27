#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Best-case approximate-hit experiment on request traces (Aker workloads).

Workloads (datasets/aker/, see download.sh):
  spacev-aker-{0.3,0.6,0.99}   SPACEV 10M (100-d int8) + Aker simZipf traces
  triviaqa-aker-{0.3,0.6,0.99} SPHERE 10M (768-d) + Aker TriviaQA paraphrase traces
  spacev-real                  SPACEV 10M + SPACEV1B historical query log
                               (train) followed by the official query set (test)

Each trace is split in time order into consecutive chunks:
  cand   candidate-generating requests (their top-K balls are cache candidates)
  score  scoring requests (greedy utility)
  fit / dev / cal   predictor fitting, model selection, calibration
  test   held-out evaluation (also the arrival order for online admission)
Repeated vectors within a chunk are deduplicated and weighted.

Exact neighbors are computed once (exact_topk_blocked) for the globally
distinct request vectors and saved to <out>/topk.npz, so every later
evaluation uses identical neighbor lists.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from simulations.approximate_hits import run_e2e_oracle_sift as base
from simulations.approximate_hits.greedy_population import GreedyPopulator
from simulations.approximate_hits.run_e2e_oracle_sift import (
    DEFAULT_K_VALUES, DEFAULT_TARGETS, exact_topk_blocked, plot_results, _print_target,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

AKER = Path("datasets/aker")
SPLITS = ("cand", "score", "fit", "dev", "cal", "test")
TRACE_FRACTIONS = {"cand": 0.08, "score": 0.32, "fit": 0.15, "dev": 0.15,
                   "cal": 0.15, "test": 0.15}
DEFAULT_CHECKPOINTS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

WORKLOADS = {
    **{f"spacev-aker-{s}": {"base": "spacev-10m.npy",
                            "trace": [f"spacev-sim-100k-{s}.npy"]}
       for s in ("0.3", "0.6", "0.99")},
    **{f"triviaqa-aker-{s}": {"base": "sphere-10m.npy",
                              "trace": [f"triviaqa-sim-100k-{s}.npy"]}
       for s in ("0.3", "0.6", "0.99")},
    "spacev-real": {"base": "spacev-10m.npy",
                    "trace": ["spacev1b/query_log.bin", "spacev1b/query.bin"]},
}

#-------------------------------------------------------------------------------


def _read_spacev_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.int32)
        return np.frombuffer(f.read(int(n) * int(d)), dtype=np.int8).reshape(n, d)


def _load_vectors(path: Path) -> np.ndarray:
    return _read_spacev_bin(path) if path.suffix == ".bin" else np.load(path)


def load_base(name: str) -> np.ndarray:
    """Database vectors: int8 kept as is; floating data as float32."""
    arr = np.load(AKER / WORKLOADS[name]["base"], mmap_mode="r")
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr)
    out = np.empty(arr.shape, dtype=np.float32)
    for b in range(0, len(arr), 500_000):
        out[b:b + 500_000] = arr[b:b + 500_000]
    return out


def split_boundaries(name: str, parts) -> dict:
    """Consecutive request ranges per split."""
    if name == "spacev-real":
        log_n, test_n = len(parts[0]), len(parts[1])
        sizes = {"cand": 10000, "score": 34162, "fit": 16000, "dev": 17000,
                 "cal": log_n - 10000 - 34162 - 16000 - 17000, "test": test_n}
    else:
        n = sum(len(p) for p in parts)
        sizes = {s: int(TRACE_FRACTIONS[s] * n) for s in SPLITS}
        sizes["test"] = n - sum(v for s, v in sizes.items() if s != "test")
    bounds, start = {}, 0
    for s in SPLITS:
        bounds[s] = (start, start + sizes[s])
        start += sizes[s]
    return bounds


def build_trace(name: str):
    """Requests (in order) as indices into globally distinct vectors."""
    parts = [_load_vectors(AKER / p) for p in WORKLOADS[name]["trace"]]
    seq = np.concatenate(parts)
    uniq, inv = np.unique(seq, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    return uniq, inv, split_boundaries(name, parts)


def split_view(inv, bounds, split):
    """Distinct global ids in a split, their weights, and the request order."""
    lo, hi = bounds[split]
    ids, local, counts = np.unique(inv[lo:hi], return_inverse=True, return_counts=True)
    return ids, counts.astype(np.int64), local.reshape(-1)


def load_trace_run(run_dir: Path):
    """Everything an evaluation script needs from a saved trace run."""
    data = json.loads((run_dir / "results.json").read_text())
    cfg = argparse.Namespace(**data["config"])
    db = load_base(cfg.trace)
    z = np.load(run_dir / "topk.npz")
    uniq, inv, topk_all = z["uniq"], z["inv"], z["topk"]
    bounds = {s: tuple(z[f"bounds_{s}"]) for s in SPLITS}
    cand_ids = split_view(inv, bounds, "cand")[0]
    sets, topk, seqs = {}, {}, {}
    for s in ("score", "fit", "dev", "cal", "test"):
        ids, w, local = split_view(inv, bounds, s)
        sets[s] = (uniq[ids], w)
        topk[s] = topk_all[ids][:, :cfg.k]
        seqs[s] = local
    return {"data": data, "cfg": cfg, "db": db, "cand_q": uniq[cand_ids],
            "cand_top": z["cand_top"], "sets": sets, "topk": topk, "seqs": seqs}


#-------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--trace", choices=sorted(WORKLOADS), required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    ap.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--checkpoints", type=float, nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--max-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.trace}...")
    db = load_base(args.trace)
    uniq, inv, bounds = build_trace(args.trace)
    logger.info(f"  db={db.shape} {db.dtype}; requests={len(inv)} distinct={len(uniq)}; "
                + ", ".join(f"{s}={hi - lo}" for s, (lo, hi) in bounds.items()))

    topk_path = out_dir / "topk.npz"
    cand_ids = split_view(inv, bounds, "cand")[0]
    k_max = max(args.k_values)
    if topk_path.exists():
        z = np.load(topk_path)
        topk_all, cand_top = z["topk"], z["cand_top"]
        logger.info(f"Loaded saved neighbors from {topk_path}")
    else:
        logger.info(f"Exact top-{k_max} for {len(cand_ids)} candidate vectors...")
        cand_top = exact_topk_blocked(uniq[cand_ids], db, k_max)
        logger.info(f"Exact top-{args.k} for {len(uniq)} distinct request vectors...")
        topk_all = exact_topk_blocked(uniq, db, args.k)
        np.savez(topk_path, uniq=uniq, inv=inv, topk=topk_all, cand_top=cand_top,
                 **{f"bounds_{s}": np.array(b) for s, b in bounds.items()})
        logger.info(f"Saved {topk_path}")

    cand_vectors = [cand_top[i, :K] for i in range(len(cand_ids)) for K in args.k_values]
    score_ids, score_w, _ = split_view(inv, bounds, "score")
    test_ids, test_w, test_seq = split_view(inv, bounds, "test")
    score_topk = topk_all[score_ids][:, :args.k]
    test_topk = topk_all[test_ids][:, :args.k]
    n_db = len(db)
    max_size = int(args.max_frac * n_db)
    checkpoints_abs = sorted(int(c * n_db) for c in args.checkpoints if c <= args.max_frac)
    logger.info(f"{len(cand_vectors)} candidates; scoring distinct {len(score_ids)}; "
                f"test distinct {len(test_ids)}")

    shared = dict(cand_vectors=cand_vectors, score_topk=score_topk, test_topk=test_topk,
                  n_db=n_db, k=args.k, max_size=max_size, checkpoints_abs=checkpoints_abs,
                  score_weights=score_w, test_weights=test_w, test_sequence=test_seq)
    logger.info("Building shared greedy index...")
    shared["template"] = GreedyPopulator(cand_vectors, score_topk, n_db, args.k, 1.0,
                                         eval_topk=test_topk, score_weights=score_w,
                                         eval_weights=test_w)
    del db  # not needed for greedy; free memory before forking workers
    import multiprocessing as mp
    base._SHARED.update(shared)
    logger.info(f"Greedy population for targets {args.targets} ({args.workers} workers)...")
    with mp.get_context("fork").Pool(min(args.workers, len(args.targets))) as pool:
        results = pool.map(base._run_target_worker, args.targets)
    for res in results:
        _print_target(res)

    config = dict(vars(args), n_db=n_db,
                  split_requests={s: hi - lo for s, (lo, hi) in bounds.items()})
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": config, "results": results}, f)
    plot_results(results, out_dir, n_db, title=args.trace)
    logger.info(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
