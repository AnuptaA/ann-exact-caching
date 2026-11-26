#!/usr/bin/env python3
"""MedRAG's working set and the cache budgets derived from it (prepare stage).

Runs ReCal's construction (no serving) on the history of the stream at increasing budgets, takes the working set
W as the smallest budget beyond which the history hit rate grows by less than one point, and writes
medrag_budgets.json: the sweep, W, the Q1 budgets 0.1 W, 0.3 W and 1.0 W and the Q3 stand-in for 5%, 0.3 W
(all as fractions of the stored bytes).

  PYTHONPATH=. .venv-core/bin/python -m scripts.medrag_working_set --prep results/online/medrag-ip \
      --backend-file results/backends/medrag/diskann.npz --coord-bytes 4 --out results/medrag/medrag_budgets.json
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recal import history_mode, sim, working_set as ws
from recal.params import Params


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--stream", default="trace")
    ap.add_argument("--backend", choices=["exact", "stored", "file"], default="file")
    ap.add_argument("--backend-file", default=None)
    ap.add_argument("--backend-target", default="0.95")
    ap.add_argument("--coord-bytes", type=float, default=4.0)
    ap.add_argument("--history-frac", type=float, default=history_mode.HISTORY_FRACTION)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--low", type=float, default=1e-5)
    ap.add_argument("--high", type=float, default=0.05)
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--min-gain", type=float, default=ws.MIN_GAIN)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    db, q, top, streams = sim.load_prep(Path(a.prep))
    idx = np.asarray(streams[a.stream], dtype=np.int64)
    history = idx[:history_mode.history_length(len(idx), a.history_frac)]
    args = SimpleNamespace(backend_file=a.backend_file, backend_target=a.backend_target, pg_dsn=None)
    backend = sim.make_backend(a.backend, db, q, top, idx, args)
    vectors = np.asarray(q[history], dtype=np.float32)
    ks = np.full(len(history), a.k, dtype=np.int32)
    params = Params(K=a.k, T=a.target, coord_bytes=a.coord_bytes, cold_start=10_000)
    budgets = ws.geometric_sweep(a.low, a.high, a.factor)
    rates = ws.sweep_rates(backend, vectors, ks, params, len(db), db.shape[1], budgets, a.coord_bytes)
    w = ws.working_set(budgets, rates, a.min_gain)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(f".{a.out.name}.tmp")
    tmp.write_text(json.dumps(ws.budgets_file(budgets, rates, w), indent=1))
    tmp.replace(a.out)
    print(f"working set {w:g} of the stored bytes; Q1 budgets {[s * w for s in ws.Q1_SHARES]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
