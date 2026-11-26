"""Checks run before any heavy work: every problem is named with the file, the expected and the actual value."""

import json
from pathlib import Path

import numpy as np

from recal.micro import world as world_mod

MIN_STREAM = 1000          # requests: below this the 30% history and the held-out part are too small to certify
NEEDS_CORE_ACCESSORS = ("intruder_check", "E10", "E11", "E13")
NEEDS_HNSWLIB = ("E7", "E13")


def check_prep(prep: Path, stream: str, max_requests: int) -> tuple[list[str], dict]:
    errors, info = [], {}
    if not prep.is_dir():
        return [f"{prep}: folder is missing; run make prepare with STAGES=fetch,prep for this dataset"], info
    for name in ("meta.json", "q.npy", "top500.npy", "streams.npz"):
        if not (prep / name).exists():
            errors.append(f"{prep / name}: file is missing; run make prepare with STAGES=fetch,prep for this dataset")
    if errors:
        return errors, info
    q = np.load(prep / "q.npy", mmap_mode="r")
    top = np.load(prep / "top500.npy", mmap_mode="r")
    streams = np.load(prep / "streams.npz")
    if stream not in streams.files:
        return [f"{prep / 'streams.npz'}: has no stream {stream!r}, only {streams.files}"], info
    idx = streams[stream]
    if top.ndim != 2 or top.shape[0] != len(q):
        errors.append(f"{prep / 'top500.npy'}: expected shape ({len(q)}, >=100) like q.npy, found {top.shape}; "
                      "rerun the prep stage")
    elif top.shape[1] < 100:
        errors.append(f"{prep / 'top500.npy'}: expected at least 100 ranks per query, found {top.shape[1]}")
    if len(idx) and (idx.min() < 0 or idx.max() >= len(q)):
        errors.append(f"{prep / 'streams.npz'}[{stream}]: query indices must lie in [0, {len(q)}), found "
                      f"[{idx.min()}, {idx.max()}]")
    n = min(len(idx), max_requests) if max_requests else len(idx)
    if n < MIN_STREAM:
        errors.append(f"{prep / 'streams.npz'}[{stream}]: expected at least {MIN_STREAM} requests, found {n}")
    info = dict(n_requests=int(n), n_queries=int(len(q)), top_ranks=int(top.shape[1]),
                history=int(0.3 * n), heldout=int(n - int(0.3 * n)))
    return errors, info


def check_modules(experiments: list[str]) -> list[str]:
    errors = []
    if any(e in NEEDS_HNSWLIB for e in experiments):
        try:
            import hnswlib  # noqa: F401
        except ImportError:
            errors.append("hnswlib is not importable (needed by E7 and E13): pip install hnswlib in .venv-core")
    if any(e in NEEDS_CORE_ACCESSORS for e in experiments):
        from recal.core import Core
        missing = [n for n in ("entries_read_batch", "shell_codes", "quantizer") if not hasattr(Core, n)]
        if missing:
            errors.append(f"the compiled core lacks {', '.join(missing)}: rebuild it with sh recal/core/build.sh")
    return errors


def probe_build(prep: Path, stream: str, budget: float, coord_bytes: float, params: dict) -> list[str]:
    """Build a small cache from a stream prefix: stops early on a broken prepare instead of after hours."""
    try:
        world = world_mod.build_world(prep, stream, budget, coord_bytes, max_requests=PROBE_REQUESTS, params=params)
    except Exception as error:  # noqa: BLE001 - reported with the context below
        return [f"the cache does not build from the first {PROBE_REQUESTS} requests of {prep} ({stream}): "
                f"{type(error).__name__}: {error}"]
    if not world.built.n_entries:
        return [f"the probe cache of {prep} ({stream}) at budget {budget:g} has no entries; check the budget"]
    return []


PROBE_REQUESTS = 12_000


def run_checks(prep: str, stream: str, experiments: list[str], max_requests: int, budget: float,
               coord_bytes: float, params: dict, probe: bool = True) -> dict:
    errors, info = check_prep(Path(prep), stream, max_requests)
    errors += check_modules(experiments)
    if not errors and probe:
        errors += probe_build(Path(prep), stream, budget, coord_bytes, params)
    if errors:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(errors))
    print(f"preflight ok: {json.dumps(info)}", flush=True)
    return info
