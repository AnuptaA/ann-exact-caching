#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Split a fixed memory budget between cache and shell sample, chosen on dev.

For each budget M (fraction of the database, counting cached vectors plus
the sample's vector-equivalents), options:
  cache only   greedy cache of size M, 6-feature balls-only model
  split f, R   greedy cache of size f * M (prefix of the greedy order) plus a
               shell sample at ranks up to R filling the rest: p = min(1,
               (1 - f) M / memory of the full shell), 7-feature model
Every option: logistic model fit on FIT, lambda on CAL (normal_fixed), hit
rate on DEV and TEST. The option with the best DEV hit rate is chosen (no
test data used); reported with its test hit rate and retention, next to every
option's test result and the oracle at M (interpolated on the greedy curve).

Writes <run>/shell_budget.json. Requires scikit-learn (.venv python).
"""

import json
import logging
from pathlib import Path


import numpy as np


from simulations.approximate_hits.common import config_of
from utils.cache.metrics import interp_log, result_for_target
from utils.cache.features import shell_block, split_inputs
from utils.cache.metrics import expand_index
from utils.cache.models import logistic_scores
from utils.cache.parallel import worker_pool
from utils.cache.shell import SAMPLE_J, load_cand_ext, shell_ids
from utils.cache.workloads import EVAL_SPLITS, load_run_data
from utils.cache.calibration import select, threshold_grid

logger = logging.getLogger(__name__)

DELTA = 0.05
BITS = 4

#-------------------------------------------------------------------------------


def cumulative_cache_size(run, selected):
    """Number of distinct cached vectors after each prefix of the greedy order."""
    seen = np.zeros(len(run.db), dtype=bool)
    out = np.empty(len(selected), dtype=np.int64)
    total = 0
    for i, j in enumerate(selected):
        ids = np.asarray(run.cached_ids[j])
        new = ids[~seen[ids]]
        seen[new] = True
        total += len(new)
        out[i] = total
    return out


def prefix_for(frac, n_db, selected, cum):
    """Longest greedy prefix whose cache is at most frac of the database."""
    n = int(np.searchsorted(cum, frac * n_db, side="right"))
    return selected[:n]


def full_shell_memory(run, bl, in_cache, R):
    shells = [shell_ids(run, i, k, R) for i, k in zip(bl["cand"], bl["K"]) if k < R]
    if not shells:
        return 0.0
    ids = np.unique(np.concatenate(shells))
    return (~in_cache[ids]).sum() * BITS / 32 / len(run.db)


def eval_option(task, run, args, selected, cum):
    budget, f, R = task
    target, k = args.target, run.k
    entries = prefix_for(f * budget, len(run.db), selected, cum)
    if not entries:
        return None
    inp = split_inputs(run, entries, EVAL_SPLITS)
    if inp is None:
        return None
    st, bl, y, w = inp.st, inp.bl, inp.y, inp.w
    X = inp.B
    cache_frac = len(st["cached"]) / len(run.db)
    p, mem = 0.0, 0.0
    if f < 1:
        full = full_shell_memory(run, bl, st["in_cache"], R)
        p = min(1.0, (budget - cache_frac) / full) if full > 0 else 0.0
        samp, S = shell_block(run, inp, p, R, BITS, [SAMPLE_J])
        mem = samp["memory"] if samp else 0.0
        X = {sp: np.hstack([X[sp], S[sp]]) for sp in EVAL_SPLITS}
    s = logistic_scores(X, y["fit"], w["fit"])
    ex = {sp: expand_index(w[sp]) for sp in EVAL_SPLITS}
    lam = select("normal_fixed", s["cal"][ex["cal"]], y["cal"][ex["cal"]],
                 threshold_grid(s["fit"]), target, DELTA)
    out = {"budget": budget, "f": f, "R": R if f < 1 else None, "p": p,
           "cache_frac": cache_frac, "sample_memory": mem}
    for sp in ("dev", "test"):
        st_, yt = s[sp][ex[sp]], y[sp][ex[sp]]
        adm = st_ >= lam if lam is not None else np.zeros(len(st_), dtype=bool)
        out[f"{sp}_hit_rate"] = float(adm.mean())
        out[f"{sp}_hit_retention"] = float(yt[adm].mean()) if adm.any() else None
    return out


#-------------------------------------------------------------------------------


def add_args(ap):
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--budgets", type=float, nargs="+", required=True)
    ap.add_argument("--fs", type=float, nargs="+", default=[0.85, 0.7])
    ap.add_argument("--Rs", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--n-each", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=6)
    ap.set_defaults(run=run)


def run(args):
    run_dir = Path(args.run_dir)
    data, _, run = load_run_data(run_dir, args.n_each)
    if max(args.Rs) > 500:
        run.cand_ext = load_cand_ext(run, run_dir, max(args.Rs))
    res = result_for_target(data, args.target)
    cum = cumulative_cache_size(run, res["selected"])
    tasks = [(b, 1.0, None) for b in args.budgets]
    tasks += [(b, f, R) for b in args.budgets for f in args.fs for R in args.Rs]
    with worker_pool(eval_option, args.workers, len(tasks), run=run, args=args,
                     selected=res["selected"], cum=cum) as pool:
        rows = [r for r in pool.map(tasks) if r is not None]
    pts = sorted(res["points"], key=lambda p: p["cache_frac"])
    xs, ys = [p["cache_frac"] for p in pts], [p["test_hit_rate"] for p in pts]
    summary = []
    for b in args.budgets:
        opts = [r for r in rows if r["budget"] == b]
        if not opts:
            continue
        best = max(opts, key=lambda r: r["dev_hit_rate"])
        only = next((r for r in opts if r["f"] == 1.0), None)
        summary.append({"budget": b, "oracle": interp_log(xs, ys, b), "chosen": best,
                        "cache_only": only, "options": opts})
        logger.info(f"budget {100 * b:.2f}%: oracle {interp_log(xs, ys, b)} | cache only "
                    f"{only['test_hit_rate'] if only else None} | chosen f={best['f']} R={best['R']} "
                    f"p={best['p']:.2f}: test {best['test_hit_rate']:.3f} "
                    f"(ret {best['test_hit_retention']})")
    (run_dir / "shell_budget.json").write_text(
        json.dumps({"args": config_of(args), "budgets": summary}, indent=1))
