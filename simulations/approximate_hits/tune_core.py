#!/usr/bin/env python3
"""Stage-1 / stage-2 search lists of the two-stage hit path, tuned on the C++ core (hnswlib) that the timing
runs use (doc 9ah; the rules of stage1_tune.py, doc 9af, which tuned FAISS graphs):
  reference   exact search over the cached vectors (the core's exact mode) on the last n warm-up requests,
              n doubled from 600 until they hold >= 200 exact hits (or the whole warm-up)
  stage 1     search list halved from 512 while hit decisions agree with exact search on >= 99% of the
              requests and the hit rate moves by <= 0.005, at every cutoff (the state's, or --lam when the
              state has none); the lowest passing search list
  stage 2     served points of the exact-hit requests: search list halved from 2048 while their mean
              retention (overlap with DiskANN's top-10) stays within 0.003 of search list 2048's
Writes results/tuning_core/<workload>/b<budget>.json.

  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/tune_core.py --workload tripclick-ip --budget 0.1
"""
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RES = Path("simulations/approximate_hits/results")
AGREE, HIT_TOL, RET_TOL, MIN_HITS = 0.99, 0.005, 0.003, 200


def core_from_state(db, q, st, shell_balls, graph=True, threads=16):
    """The warm-up cache of a saved state in the core (union of the balls as cached vectors; entries with
    their sampled shells, as stored)."""
    from simulations.approximate_hits.cachecore import Core
    core = Core(db.shape[1], int(st["k"]), shell_balls, capacity=len(st["cached_ids"]) + 1024, graph=graph)
    core.set_threads(threads)
    ids = np.asarray(st["cached_ids"], dtype=np.int64)
    core.add_cached(ids, np.ascontiguousarray(np.asarray(db[ids], dtype=np.float32)))
    core.set_model(list(map(float, st["mu"])), list(map(float, st["sd"])), list(map(float, st["coef"])),
                   float(st["intercept"]))
    sv, sid = st["sv"].astype(np.float32), st["samp_ids"].astype(np.int64)
    if int(st["bits"]) == 4 and len(sv):             # packed codes, the state's range (doc 9ai)
        core.set_quantizer(list(map(float, sv.min(axis=0))), list(map(float, sv.max(axis=0))), 4)
    es, eo = st["entry_samp"], st["entry_samp_off"]
    ei, ev = np.empty(0, np.int64), np.empty((0, db.shape[1]), np.float32)
    for j, (c, r) in enumerate(zip(st["centers"], st["radius"])):
        pos = es[eo[j]:eo[j + 1]]
        core.add_entry(int(c), np.asarray(q[c], np.float32), float(r), ei, ev, sid[pos], sv[pos])
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--lam", default=None, help="comma-separated cutoffs (cells whose state has none)")
    ap.add_argument("--shell-balls", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from simulations.approximate_hits.workloads import get
    from simulations.approximate_hits.prep_io import load_prep
    w = get(a.workload)
    db, q, _, _ = load_prep(w.prep_dir)
    st = dict(np.load(w.state_dir / f"state_b{a.budget:g}.npz", allow_pickle=False))
    ans = np.load(w.answers_file)[f"answer_{w.target}"][:, :10]
    warm_all = np.asarray(np.load(w.prep_dir / "streams.npz")[w.stream][:int(st["warmup"])], dtype=np.int64)
    lams = [float(v) for v in a.lam.split(",")] if a.lam else [float(st["lam"])]
    if not all(np.isfinite(lams)):
        raise SystemExit("the state has no cutoff: pass --lam (cutoffs the recalibrating controllers use)")
    lam = min(lams)
    core = core_from_state(db, q, st, a.shell_balls, threads=a.threads)

    def run(xs, l_, ef1, ef2=0):
        hit, score, near, nr, feat = core.lookup_batch(xs, l_, ef1, ef2)[:5]
        return np.asarray(hit), np.asarray(score), np.asarray(near)

    n = 600
    while True:
        req = warm_all[-n:]
        xs = np.ascontiguousarray(np.asarray(q[req], np.float32))
        _, s_ex, near_ex = run(xs, lam, 0)
        hit_ex = s_ex >= lam
        if hit_ex.sum() >= MIN_HITS or n >= len(warm_all):
            break
        n = min(2 * n, len(warm_all))
    res = {"workload": w.name, "budget": a.budget, "n_requests": int(len(req)), "n_exact_hits": int(hit_ex.sum()),
           "exact_hit_rate": float(hit_ex.mean()), "cutoffs": lams, "lib": "hnswlib (cachecore)",
           "stage1_trials": [], "stage2_trials": []}
    logger.info(f"{w.name} b{a.budget:g}: {core.n_cached()} cached vectors, {len(req)} requests, {int(hit_ex.sum())} exact hits")
    best1 = None
    for ef in (512, 256, 128, 64, 32, 16):
        _, s, _ = run(xs, lam, ef)
        agree, dh = 1.0, 0.0
        for l_ in lams:
            h, he = s >= l_, s_ex >= l_
            a_, d_ = float((h == he).mean()), float(h.mean() - he.mean())
            agree, dh = min(agree, a_), (dh if abs(dh) >= abs(d_) else d_)
        core.set_threads(1)
        t0 = time.perf_counter()
        for x in xs[:300]:
            core.lookup(x, lam, ef, 0)
        ms = (time.perf_counter() - t0) / min(300, len(xs)) * 1e3
        core.set_threads(a.threads)
        ok = agree >= AGREE and abs(dh) <= HIT_TOL
        res["stage1_trials"].append({"ef": ef, "agree": agree, "dhit": dh, "ms_lookup": ms, "ok": ok})
        logger.info(f"  stage 1 ef {ef}: agree {agree:.4f} dhit {dh:+.4f} lookup {ms:.3f} ms -> {'ok' if ok else 'FAIL'}")
        if not ok:
            break
        best1 = ef
    if best1 is None:
        raise SystemExit("stage 1: even search list 512 disagrees with exact search")
    res["stage1"] = {"ef": best1}
    ref = None
    best2 = None
    for ef2 in (2048, 1024, 512, 256, 128, 64, 32):
        _, _, near = run(xs, float("-inf"), best1, ef2)      # every request takes the stage-2 path
        ov = np.array([np.isin(ans[qi], nn).mean() for qi, nn in zip(req, near)])
        r_ = float(ov[hit_ex].mean())
        ref = r_ if ref is None else ref
        ok = ref - r_ <= RET_TOL
        res["stage2_trials"].append({"ef": ef2, "retention": r_, "gap_vs_2048": ref - r_, "ok": ok})
        logger.info(f"  stage 2 ef {ef2}: retention of exact hits {r_:.4f} (2048: {ref:.4f}) -> {'ok' if ok else 'FAIL'}")
        if not ok:
            break
        best2 = ef2
    res["stage2"] = {"ef": best2}
    exact_ret = float(np.array([np.isin(ans[qi], nn).mean() for qi, nn in zip(req, near_ex)])[hit_ex].mean())
    res["stage2"]["exact_retention"] = exact_ret
    out = Path(a.out) if a.out else RES / "tuning_core" / w.name / f"b{a.budget:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    logger.info(f"stage 1 {best1}, stage 2 {best2} -> {out}")


if __name__ == "__main__":
    main()
