#!/usr/bin/env python3
"""Extra count columns for a hit table (doc 9al): the intruder count over the whole sample at p0 with
8-bit codes and with 4-bit codes over a robust range (1st / 99th percentile per dimension instead of
min / max), to see whether the 4-bit codes' loss vs float (ladder) is the code width or the range.
Rebuilds the same warm-up cache (deterministic) and reuses the table's requests and radii.
Writes <table>.aug.npz; hit_ladder.py picks it up.

  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/hit_table_aug.py results/hit_tables/<w>/<t>.npz
"""
import sys
import json
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

from simulations.approximate_hits import hit_table as H
from simulations.approximate_hits.workloads import get

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def codes(x, lo, hi, bits):
    step = np.maximum(hi - lo, 1e-12) / (2 ** bits - 1)
    return (lo + np.round((np.clip(x, lo, hi) - lo) / step) * step).astype(np.float32)


def main():
    path = Path(sys.argv[1])
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    OS, args, D, stream, sim = H.build_cache(get(m["workload"]), m["budget"], m["variant"], m["backend"])
    cache = sim.cache
    cache.prepare()
    if abs(cache.p - m["p0"]) > 1e-12 or len(cache.samp_ids) != m["samples_p0"]:
        raise SystemExit(f"cache differs from the table: p {cache.p} vs {m['p0']}")
    U, nr = z["U"], z["nr"]
    x0 = D["db"][cache.samp_ids].astype(np.float64)
    out = {}
    if len(x0):
        lo, hi = x0.min(axis=0), x0.max(axis=0)
        rlo, rhi = np.percentile(x0, 1, axis=0), np.percentile(x0, 99, axis=0)
        sets = {"all_8": codes(x0, lo, hi, 8), "all_4r": codes(x0, rlo, rhi, 4), "all_8r": codes(x0, rlo, rhi, 8)}
        sq = {k: (v.astype(np.float64) ** 2).sum(axis=1).astype(np.float32) for k, v in sets.items()}
        for k in sets:
            out[k] = np.zeros(len(U), np.int32)
        Q = D["q"]
        for s in range(0, len(U), H.CHUNK):
            x = np.ascontiguousarray(Q[U[s:s + H.CHUNK]], dtype=np.float32)
            x_sq = (x.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
            nr2 = (nr[s:s + H.CHUNK] ** 2)[:, None]
            for k, v in sets.items():
                out[k][s:s + len(x)] = (H.sqd(x, x_sq, v, sq[k]) < nr2).sum(axis=1)
    else:
        for k in ("all_8", "all_4r", "all_8r"):
            out[k] = np.zeros(len(U), np.int32)
    np.savez(path.with_suffix(".aug.npz"), **out)
    logger.info(f"wrote {path.with_suffix('.aug.npz')}")


if __name__ == "__main__":
    main()
