#!/usr/bin/env python3
"""Smoke test of a freshly set up machine: a small synthetic cell per ReCal mode, then library checks.

  python -m scripts.smoke_cells --out <dir> [--skip-libs]

The cells run recal.cells on a synthetic prep (default mode, idealized, every ablation switch). The library
checks load the Aker shims, diskannpy and the pgvector extension; they are skipped when --skip-libs is given
and reported as MISSING otherwise.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SHIM_DIR = Path("simulations/approximate_hits/aker_real")
ABLATIONS = ("--no-union", "--no-expansion", "--no-shells", "--lru", "--no-calibration", "--no-recalibration",
             "--no-mini-cache")


def make_prep(path: Path, n: int = 4000, d: int = 12, requests: int = 3600, seed: int = 0) -> Path:
    from recal.backend import ExactBackend
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(40, d))
    data = (centers[rng.integers(0, 40, n)] + rng.normal(size=(n, d))).astype(np.float32)
    queries = (centers[rng.integers(0, 40, 1500)] + rng.normal(size=(1500, d))).astype(np.float32)
    stream = rng.integers(0, len(queries), requests)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "db.npy", data)
    np.save(path / "q.npy", queries)
    np.save(path / "top500.npy", ExactBackend(data).search(queries, 100)[0].astype(np.int32))
    np.savez(path / "streams.npz", random=stream.astype(np.int64))
    (path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(path / "db.npy"),
                                                "q": str(path / "q.npy")}))
    return path


def run_cell(prep: Path, out: Path, flags: list[str]) -> dict:
    cmd = [sys.executable, "-m", "recal.cells", "--method", "recal", "--prep", str(prep), "--stream", "random",
           "--budget", "0.6", "--k", "5", "--backend", "exact", "--history-frac", "0.3", "--history", "3000",
           "--min-ess", "30", "--target", "0.5", "--out", str(out), *flags]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "."}, capture_output=True)
    result = json.loads((out / "recal.json").read_text())
    if not result["within_M"] or not (out / "recal_records.npz").exists():
        raise RuntimeError(f"cell {flags} left M or wrote no records")
    return result


def check_libraries() -> dict:
    status = {}
    for name in ("libakershim.so", "libakershim_prox.so"):
        try:
            ctypes.CDLL(str((SHIM_DIR / name).resolve()))
            status[name] = "ok"
        except OSError as e:
            status[name] = f"MISSING ({e})"
    try:
        import diskannpy  # noqa: F401
        status["diskannpy"] = "ok"
    except ImportError as e:
        status["diskannpy"] = f"MISSING ({e})"
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-libs", action="store_true")
    a = ap.parse_args(argv)
    prep = make_prep(a.out / "prep")
    failures = 0
    for label, flags in [("default", []), ("idealized", ["--idealized"])] + [(f.lstrip("-"), [f]) for f in ABLATIONS]:
        try:
            r = run_cell(prep, a.out / label, flags)
            print(f"cell {label}: ok, scored {r['n_scored']}, constructions {r['constructions']}")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"cell {label}: FAIL {e}")
            failures += 1
    if not a.skip_libs:
        for name, state in check_libraries().items():
            print(f"{name}: {state}")
            failures += state.startswith("MISSING")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
