#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Larger shell samples, and how many calibration queries lambda needs.

Per cache checkpoint of one saved run (greedy cache for --target), the
7-feature logistic model (3 balls + intruder count, eval_final) with shell
samples at every (p, R) in --ps x --Rs (R > 500 uses exact top-R neighbor
lists of the candidate queries, computed once and saved):
  test hit rate, realized retention, sample memory
Calibration-size study, default shell (p = 0.25, R = 500): the cal, dev and
test requests are pooled and re-split at random --draws times per n: n
requests calibrate lambda (each method in --methods), the rest evaluate it.
Reports mean hit rate and the fraction of draws whose admitted mean
retention on the evaluation part is below T and below T - 0.01.

Writes <run>/shell_size.json. Requires scikit-learn (.venv python).
"""

import json
import argparse
import logging
from pathlib import Path


import numpy as np

from utils.base.logs import setup_logging

from utils.cache.features import shell_block, split_inputs
from utils.cache.metrics import admit, expand_index, result_for_target, state_tasks
from utils.cache.models import logistic_scores
from utils.cache.parallel import worker_pool
from utils.cache.shell import SAMPLE_J, load_cand_ext
from utils.cache.workloads import EVAL_SPLITS, load_run_data
from utils.cache.calibration import select, threshold_grid

logger = logging.getLogger(__name__)

DELTA = 0.05
BITS = 4
BASE = (0.25, 500)

#-------------------------------------------------------------------------------


def cal_study(s_pool, y_pool, grid, target, ns, draws, methods, rng):
    """Repeated random re-splits of the pooled cal + dev + test requests: n
    requests calibrate lambda, all the others evaluate it (fresh, independent
    calibration sets, so the below-T frequency estimates the failure rate)."""
    out = {m: {} for m in methods}
    for n in ns:
        if n >= len(s_pool):
            continue
        res = {m: ([], 0, 0) for m in methods}
        for _ in range(draws):
            perm = rng.permutation(len(s_pool))
            c, e = perm[:n], perm[n:]
            for m in methods:
                lam = select(m, s_pool[c], y_pool[c], grid, target, DELTA)
                h, r = admit(lam, s_pool[e], y_pool[e])
                hits, below, below01 = res[m]
                hits.append(h)
                res[m] = (hits, below + (r is not None and r < target),
                          below01 + (r is not None and r < target - 0.01))
        for m in methods:
            hits, below, below01 = res[m]
            out[m][str(n)] = {"mean_hit_rate": float(np.mean(hits)),
                              "frac_below_T": below / draws, "frac_below_T_0.01": below01 / draws}
    return out


def eval_point(task, run, args):
    target, point, entries = task
    inp = split_inputs(run, entries, EVAL_SPLITS)
    if inp is None:
        return None
    sets, st, bl, y, w, nr, B = run.sets, inp.st, inp.bl, inp.y, inp.w, inp.nr, inp.B
    ex = {sp: expand_index(w[sp]) for sp in EVAL_SPLITS}
    out = {"cache_frac": point["cache_frac"], "oracle_hit_rate": point["test_hit_rate"],
           "shells": {}}
    for R in args.Rs:
        for p in args.ps:
            samp, S = shell_block(run, inp, p, R, BITS, [SAMPLE_J])
            X = {sp: np.hstack([B[sp], S[sp]]) for sp in EVAL_SPLITS}
            s = logistic_scores(X, y["fit"], w["fit"])
            grid = threshold_grid(s["fit"])
            lam = select("normal_fixed", s["cal"][ex["cal"]], y["cal"][ex["cal"]], grid,
                         target, DELTA)
            h, r = admit(lam, s["test"][ex["test"]], y["test"][ex["test"]])
            out["shells"][f"p={p:g},R={R}"] = {"hit_rate": h, "hit_retention": r,
                                               "sample_memory": samp["memory"] if samp else 0.0}
            if (p, R) == BASE:
                pool = ("cal", "dev", "test")
                s_pool = np.concatenate([s[sp][ex[sp]] for sp in pool])
                y_pool = np.concatenate([y[sp][ex[sp]] for sp in pool])
                out["cal_study"] = cal_study(s_pool, y_pool, grid, target, args.cal_sizes,
                                             args.draws, args.methods, np.random.default_rng(0))
    return out


#-------------------------------------------------------------------------------


def main():
    setup_logging()
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--sizes", type=float, nargs="+", required=True)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--Rs", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--cal-sizes", type=int, nargs="+", default=[250, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--methods", nargs="+",
                    default=["normal_fixed", "normal_fixed_min30", "eb_fixed"])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-name", type=str, default="shell_size.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    data, _, run = load_run_data(run_dir, args.n_each)
    if max(args.Rs) > 500:
        run.cand_ext = load_cand_ext(run, run_dir, max(args.Rs))
    res = result_for_target(data, args.target)
    tasks = state_tasks(res, args.sizes, args.target)
    out = []
    with worker_pool(eval_point, args.workers, len(tasks), run=run, args=args) as pool:
        for s in pool.imap(tasks):
            if s is None:
                continue
            out.append(s)
            logger.info(f"cache {100 * s['cache_frac']:.2f}% (oracle {s['oracle_hit_rate']:.3f}): "
                        + " ".join(f"{n} {v['hit_rate']:.3f}(+{100 * v['sample_memory']:.2f}%)"
                                   for n, v in s["shells"].items()))
            (run_dir / args.out_name).write_text(
                json.dumps({"args": vars(args), "states": out}, indent=1))


if __name__ == "__main__":
    main()
