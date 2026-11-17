#!/usr/bin/env python3
"""Parity of the C++ core with the current Python hit path on a saved warm-up state (doc 9ah):
  exact     core exact kNN vs eval_timing.HitPath (brute force, whole shell sample) and vs FastHitPath
            "brute+balls10": scores and decisions must agree (float rounding only)
  graph     core hnswlib (stage 1 at --ef) vs FastHitPath "hnsw<ef>+balls10" (FAISS): decision agreement

  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/cachecore/parity_static.py \
      --prep spacev-aker-0.6 --budget 0.01
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
import numpy as np

RES = Path("simulations/approximate_hits/results")


def core_from_state(db, q, st, shell_balls):
    from simulations.approximate_hits.cachecore import Core
    k = int(st["k"])
    core = Core(db.shape[1], k, shell_balls, capacity=len(st["cached_ids"]) + 1024)
    core.set_threads(16)
    ids = np.asarray(st["cached_ids"], dtype=np.int64)
    core.add_cached(ids, np.asarray(db[ids], dtype=np.float32))      # the union of the balls
    core.set_threads(1)
    core.set_model(list(map(float, st["mu"])), list(map(float, st["sd"])), list(map(float, st["coef"])),
                   float(st["intercept"]))
    sv, sid, es, eo = st["sv"].astype(np.float32), st["samp_ids"].astype(np.int64), st["entry_samp"], st["entry_samp_off"]
    empty_i, empty_v = np.empty(0, np.int64), np.empty((0, db.shape[1]), np.float32)
    for j, (c, r) in enumerate(zip(st["centers"], st["radius"])):
        pos = es[eo[j]:eo[j + 1]]
        core.add_entry(int(c), np.asarray(q[c], np.float32), float(r), empty_i, empty_v, sid[pos], sv[pos])
    if shell_balls == 0:          # whole-sample scope: every sample of the state, including any not in an entry
        core.add_samples(sid, sv)
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="spacev-aker-0.6")
    ap.add_argument("--budget", default="0.01")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from simulations.approximate_hits.prep_io import load_prep
    from simulations.approximate_hits.eval_timing import HitPath, FastHitPath
    db, q, _, _ = load_prep(RES / "online" / a.prep)
    st = dict(np.load(RES / "timing_v4" / a.prep / f"state_b{a.budget}.npz", allow_pickle=False))
    lam = float(st["lam"])
    if not np.isfinite(lam):
        lam = float(np.nanmedian(st["test_scores"]))
    stream = np.load(RES / "online" / a.prep / "streams.npz")["trace" if not a.prep.startswith("esci") else "random"]
    w = int(st["warmup"])
    req = np.asarray(stream[w:w + a.n], dtype=np.int64)
    xs = np.ascontiguousarray(np.asarray(q[req], np.float32))
    out = {"prep": a.prep, "budget": a.budget, "n": len(req), "lam": lam}
    T = {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0}

    def py_path(hp):
        hp.lam = lam
        s, h = np.empty(len(xs)), np.zeros(len(xs), bool)
        t0 = time.perf_counter()
        for i, x in enumerate(xs):
            h[i], s[i], _ = hp.decide(x, T)
        return s, h, (time.perf_counter() - t0) / len(xs) * 1e3

    def core_path(core, ef):
        s, h = np.empty(len(xs)), np.zeros(len(xs), bool)
        t0 = time.perf_counter()
        for i, x in enumerate(xs):
            r = core.lookup(x, lam, ef)
            s[i], h[i] = r.score, r.hit
        return s, h, (time.perf_counter() - t0) / len(xs) * 1e3

    cases = [("exact, whole sample", HitPath(db, q, st), 0, 0),
             ("exact, 10 best balls", FastHitPath(db, q, st, "brute", "balls10"), 10, 0),
             (f"graph ef {a.ef}, 10 best balls", FastHitPath(db, q, st, f"hnsw{a.ef}", "balls10"), 10, a.ef)]
    for name, hp, sb, ef in cases:
        s_py, h_py, ms_py = py_path(hp)
        core = core_from_state(db, q, st, sb)
        s_c, h_c, ms_c = core_path(core, ef)
        row = {"case": name, "python_ms": ms_py, "core_ms": ms_c, "decisions_agree": float((h_py == h_c).mean()),
               "hit_rate_python": float(h_py.mean()), "hit_rate_core": float(h_c.mean()),
               "max_abs_score_diff": float(np.max(np.abs(s_py - s_c))),
               "median_abs_score_diff": float(np.median(np.abs(s_py - s_c)))}
        out[name] = row
        print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
