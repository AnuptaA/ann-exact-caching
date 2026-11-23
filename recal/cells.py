#!/usr/bin/env python3
"""One experiment cell for any compared system, at an equal byte budget M.

M is a percentage of the dataset's stored bytes. ReCal charges its own structures (recal.sim). The earlier
simulators count the budget in vectors; here the pool is chosen so that the system's bytes by structure
(recal.accounting) stay within M: a run is repeated with a rescaled pool when it used more than M.

  python -m recal.cells --method recal|prototype|aker|aker_sim|qvcache|proximity --prep <prep> --stream random \\
      --budget 0.05 --out <dir>

aker and proximity run the real vendored Aker library (aker_real/eval_aker_real.py, paper settings, pool sized
to M from the library's own counts); proximity takes --prox-percentile of the history's pairwise squared
distances (0.1 and 1). aker_sim, prototype and qvcache are the simulators of
simulations/approximate_hits/online_baselines.py (QVCache: D 0.075, alpha 0.9, 4 mini-indexes). The real library
needs the Linux build (needs cluster check): without libakershim.so the cell stops and says so.
"""

import os
import json
import subprocess
import sys
import argparse
from pathlib import Path

import numpy as np

from recal import accounting, history_mode, sim
from recal.measure.jobs_util import register_args, stamp_result
from recal.records import Records

SIM = "simulations/approximate_hits/online_sim.py"
SIM_METHOD = {"prototype": "static", "aker_sim": "aker", "qvcache": "qvcache"}
REAL_DRIVER = "simulations/approximate_hits/aker_real/eval_aker_real.py"
REAL_METHODS = ("aker", "proximity")
REAL_SHIM = Path("simulations/approximate_hits/aker_real/libakershim.so")
PROX_SHIM = Path("simulations/approximate_hits/aker_real/libakershim_prox.so")


def counters(method: str, log: dict, d: int, args: dict, coord_bytes: float = 4) -> dict:
    cached, entries = log["cached"], log["entries"]
    if method in ("aker", "aker_sim"):
        return accounting.aker_bytes(cached, entries, args.get("k", 10), args.get("aker_reserve", 0), d,
                                     coord_bytes=coord_bytes)
    if method == "qvcache":
        return accounting.qvcache_bytes(cached, d, coord_bytes=coord_bytes)
    shell = max(0.0, (log["memory"] - cached) * 32.0 / args.get("bits", 4))
    return accounting.prototype_bytes(cached, int(shell), entries, max(cached / max(entries, 1), 1.0), d,
                                      args.get("bits", 4), coord_bytes=coord_bytes)


def stream_length(a) -> int:
    streams = sim.load_prep(Path(a.prep))[3]
    n = len(streams[a.stream])
    return min(n, a.max_requests) if a.max_requests else n


def run_simulator(method: str, a, fraction: float, tag: str) -> Path:
    cmd = [sys.executable, SIM, "--prep", a.prep, "--stream", a.stream, "--method", SIM_METHOD[method],
           "--budget", f"{fraction:.6g}", "--k", str(a.k), "--out", a.out, "--tag", tag,
           "--records"]
    if a.max_requests:
        cmd += ["--max-queries", str(a.max_requests)]
    if a.history_frac:
        cmd += ["--warmup", str(history_mode.history_length(stream_length(a), a.history_frac))]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "."})
    return Path(a.out) / f"{SIM_METHOD[method]}{tag}.json"


def next_fraction(fraction: float, used: float, M: float, lo: float | None, hi: float | None) -> float:
    """Next pool fraction: scale by M / used, kept strictly between the largest feasible and the smallest
    infeasible fraction seen so far (bytes are not monotone in the pool)."""
    nxt = fraction * M / used * (0.99 if used > M else 0.99)
    if hi is not None and nxt >= hi:
        nxt = (lo + hi) / 2 if lo is not None else hi * 0.95
    if lo is not None and nxt <= lo:
        nxt = (lo + hi) / 2 if hi is not None else lo * 1.05
    return nxt


def run_baseline(method: str, a, attempts: int = 6) -> dict:
    db, *_ = sim.load_prep(Path(a.prep))
    d, n = db.shape[1], len(db)
    M = accounting.budget_bytes(n, d, a.budget, a.coord_bytes)
    fraction = a.budget * a.coord_bytes * d / (a.coord_bytes * d + 8 + 40)        # first guess: the per-vector charge plus a share of entries
    runs, best, lo, hi = [], None, None, None
    sim_name = SIM_METHOD[method]
    for attempt in range(attempts):
        path = run_simulator(method, a, fraction, f"{a.tag}_try{attempt}")
        res = json.loads(path.read_text())
        used = counters(method, res["log"][-1], d, res.get("args", {}), a.coord_bytes)
        runs.append({"vector_fraction": fraction, "total_bytes": used["total"]})
        if used["total"] <= M:
            lo = fraction if lo is None else max(lo, fraction)
            if best is None or used["total"] > best[0]["total"]:
                best = (used, res, fraction, f"{a.tag}_try{attempt}")
            if used["total"] >= 0.95 * M:
                break
        else:
            hi = fraction if hi is None else min(hi, fraction)
        fraction = next_fraction(fraction, used["total"], M, lo, hi)
    used, res, fraction, best_tag = best if best is not None else (
        used, res, fraction, f"{a.tag}_try{len(runs) - 1}")
    kept = Path(a.out) / f"{sim_name}{best_tag}_records.npz"
    if kept.exists():
        kept.replace(Path(a.out) / f"{method}{a.tag}_records.npz")
    n = stream_length(a)
    history_mode.stamp(res, n, history_mode.history_length(n, a.history_frac))
    res.update({"bytes": used, "M": M, "within_M": bool(used["total"] <= M), "byte_runs": runs,
                "vector_fraction": fraction, "method": method})
    for p in list(Path(a.out).glob(f"*{a.tag}_try*.json")) + list(Path(a.out).glob(f"*{a.tag}_try*_records.npz")):
        p.unlink()
    return res


def real_aker_command(a, scale: float, out_file: Path) -> list[str]:
    """Command of the real-library driver for one attempt; Proximity adds the percentile threshold."""
    cmd = [a.ann_python, REAL_DRIVER, "--prep", a.prep, "--stream", a.stream, "--budget", f"{a.budget:.6g}",
           "--k", str(a.k), "--equal-bytes", "--coord-bytes", f"{a.coord_bytes:g}", "--pool-scale", f"{scale:.6g}",
           "--history-frac", f"{a.history_frac:g}", "--out", str(out_file)]
    if a.max_requests:
        cmd += ["--max-requests", str(a.max_requests)]
    if a.backend == "file":
        cmd += ["--backend-file", a.backend_file, "--backend-target", a.backend_target]
    if a.updates:
        cmd += ["--data-change", "churn"]
    if a.method == "proximity":
        cmd += ["--prox-percentile", f"{a.prox_percentile:g}"]
    return cmd


def run_real_aker(a, attempts: int = 3) -> dict:
    """Real Aker or Proximity at M: the pool is rescaled and the run repeated while it exceeds M."""
    shim = PROX_SHIM if a.method == "proximity" else REAL_SHIM
    if not shim.exists():
        raise SystemExit(f"{shim} is missing: build the Aker shim (aker_real/build.sh); needs cluster check")
    scale, res = 1.0, None
    for attempt in range(attempts):
        out_file = Path(a.out) / f"{a.method}{a.tag}_try{attempt}.json"
        subprocess.run(real_aker_command(a, scale, out_file), check=True, env={**os.environ, "PYTHONPATH": "."})
        res = json.loads(out_file.read_text())
        records = out_file.with_name(f"{out_file.stem}_records.npz")
        if res.get("within_M"):
            break
        scale *= 0.97 * res["M"] / res["bytes"]["total"]
    final = Path(a.out) / f"{a.method}{a.tag}_records.npz"
    if records.exists():
        records.replace(final)
    for extra in list(Path(a.out).glob(f"{a.method}{a.tag}_try*.json")) + list(Path(a.out).glob("*_try*_records.npz")):
        extra.unlink()
    res.update({"method": a.method, "pool_scale": scale})
    return res


def main() -> None:
    ap = sim.parser()
    ap.add_argument("--method", choices=["recal", "prototype", "aker", "aker_sim", "qvcache", "proximity"],
                    required=True)
    ap.add_argument("--prox-percentile", type=float, default=0.1,
                    help="Proximity: threshold at this percentile of squared distances between history queries")
    ap.add_argument("--ann-python", default=".venv-ann/bin/python", help="interpreter of the real-library driver")
    a = ap.parse_args()
    register_args(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.method == "recal":
        records = Records()
        result = sim.run(a, records)
        records.save(out / f"recal{a.tag}_records.npz")
    elif a.method in REAL_METHODS:
        out.mkdir(parents=True, exist_ok=True)
        result = run_real_aker(a)
    else:
        result = run_baseline(a.method, a)
    stamp_result(result)
    records_name = f"{a.method}{a.tag}_records.npz"
    if (out / records_name).exists():
        result["records_file"] = records_name
    tmp = out / f".{a.method}{a.tag}.json.tmp"
    tmp.write_text(json.dumps(result, default=str))
    tmp.replace(out / f"{a.method}{a.tag}.json")


if __name__ == "__main__":
    main()
