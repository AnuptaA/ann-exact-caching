#!/usr/bin/env python3
"""Measure the resident memory of the QVCache artifact per cached point and compare it with the byte model.

  python -m scripts.measure_qvcache_memory --base datasets/aker/spacev-10m.npy --coord-bytes 1 \\
      --sizes 25000 50000 100000 200000 --out measurement.json [--update-constants]

For each size the artifact's cache is built in a fresh process (an allocator keeps freed memory), warmed with
misses until its mini-indexes hold the size, and the resident memory above the empty cache is read. Bytes are fitted
as a x points + b. The model charges coord_bytes x d + 4 + 4 x R per point (772 for d = 128 float32, R = 64; the
paper's Table 2 implies about 990). When a differs from the model by more than 5%, --update-constants writes the
measured a and b to recal/data/qvcache_memory.json for this (d, coord_bytes, R), and every later QVCache result
records the source. Needs the built artifact (Linux); the fitting is tested on synthetic measurements.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from recal import accounting

TOLERANCE = 0.05


def rss_bytes() -> int:
    """Resident set size of this process."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)


def fit_linear(points, extra_bytes) -> dict:
    """Least squares of extra bytes against points: slope a (bytes per point), intercept b (fixed bytes)."""
    x = np.asarray(points, dtype=np.float64)
    y = np.asarray(extra_bytes, dtype=np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ np.array([a, b])
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"per_point": float(a), "fixed": float(b), "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0}


def compare(fit: dict, model_per_point: float, tolerance: float = TOLERANCE) -> dict:
    diff = (fit["per_point"] - model_per_point) / model_per_point
    return {"model_per_point": model_per_point, "measured_per_point": fit["per_point"], "relative_difference": diff,
            "replace": abs(diff) > tolerance}


def update_constants(path: Path, d: int, coord_bytes: float, R: int, fit: dict, source: str) -> dict:
    path = Path(path)
    entries = json.loads(path.read_text()) if path.exists() else {}
    entries[accounting.qvcache_memory_key(d, coord_bytes, R)] = {
        "per_point": fit["per_point"], "fixed": max(fit["fixed"], 0.0), "r2": fit["r2"], "source": source}
    path.write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n")
    return entries


def measure_one(qvc, base, size: int, conf: dict, deviation: float, workdir: Path, k: int = 10, rss=rss_bytes,
                settle_s: float = 2.0, max_wait_s: float = 900.0) -> dict:
    """Resident memory above the empty cache once the mini-indexes hold `size` points."""
    from recal import qvcache_driver as qd
    from recal.backend import ExactBackend
    from recal.qvcache_backends import StoredBackend
    workdir = Path(workdir)
    bin_path = qd.pca_base(base, workdir, min(len(base), 1_000_000))
    backend = StoredBackend(ExactBackend(np.asarray(base, dtype=np.float32)))
    cache = qd.make_cache(qvc, bin_path, workdir / "pca", backend, conf, deviation, size)
    gc.collect()
    empty = rss()
    order = np.random.default_rng(0).permutation(len(base))
    began, last, stable_since = time.time(), -1, time.time()
    for qi in order:
        cache.search(np.asarray(base[qi], dtype=np.float32), k)
        n = int(cache.get_number_of_vectors_in_memory_index())
        if n >= 0.98 * size:
            break
        if time.time() - began > max_wait_s:
            break
    while time.time() - stable_since < settle_s:        # asynchronous inserts settle
        n = int(cache.get_number_of_vectors_in_memory_index())
        if n != last:
            last, stable_since = n, time.time()
        time.sleep(0.2)
    gc.collect()
    points = int(cache.get_number_of_vectors_in_memory_index())
    return {"size": size, "points": points, "rss_empty": empty, "rss_filled": rss(), "extra": rss() - empty}


def measure_sizes(sizes, runner) -> list[dict]:
    """runner(size) -> measure_one result; each size runs in its own process in main()."""
    return [runner(int(s)) for s in sizes]


def subprocess_runner(args):
    def run(size: int) -> dict:
        cmd = [sys.executable, "-m", "scripts.measure_qvcache_memory", "--one-size", str(size), "--base", str(args.base),
               "--coord-bytes", f"{args.coord_bytes:g}", "--workdir", str(args.workdir)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": "."})
        return json.loads(out.stdout.strip().splitlines()[-1])
    return run


def report(results: list[dict], d: int, coord_bytes: float, R: int) -> dict:
    fit = fit_linear([r["points"] for r in results], [r["extra"] for r in results])
    model = accounting.qvcache_unit_costs(d, coord_bytes, R)["per_point"]
    return {"dim": d, "coord_bytes": coord_bytes, "R": R, "measurements": results, "fit": fit,
            "comparison": compare(fit, model)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--coord-bytes", type=float, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[25_000, 50_000, 100_000, 200_000])
    ap.add_argument("--one-size", type=int, help="internal: measure one size and print a JSON line")
    ap.add_argument("--workdir", type=Path, default=Path("qvcache_memory_work"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--update-constants", action="store_true")
    ap.add_argument("--constants", type=Path, default=accounting.QVCACHE_MEMORY_FILE)
    a = ap.parse_args(argv)
    from recal import qvcache_driver as qd
    base = np.load(a.base, mmap_mode="r")
    conf = qd.settings()
    if a.one_size:
        print(json.dumps(measure_one(qd.load_module(), base, a.one_size, conf, qd.default_deviation(a.coord_bytes),
                                     a.workdir)))
        return 0
    result = report(measure_sizes(a.sizes, subprocess_runner(a)), base.shape[1], a.coord_bytes, conf["R"])
    result["replaced"] = False
    if result["comparison"]["replace"] and a.update_constants:
        update_constants(a.constants, base.shape[1], a.coord_bytes, conf["R"], result["fit"],
                         f"measured on {a.base.name}, sizes {a.sizes}")
        result["replaced"] = True
    if a.out:
        a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["comparison"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
