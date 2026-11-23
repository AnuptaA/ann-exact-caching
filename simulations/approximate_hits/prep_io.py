"""Loading a prep's database, queries and exact top-500 for the timing harnesses (doc 9r).

Two kinds of prep: "trace" (SPACEV / TriviaQA Zipf traces and the SPACEV real log: meta.json
names the trace; database via run_e2e_oracle_trace, queries in q.npy) and "esci" (no meta.json:
the 1M-product ESCI database and its 130,652 queries from datasets/esci/data_vectors, as
online_sim.py and eval_real_backend.py load them)."""

import json
from pathlib import Path

import numpy as np

ESCI = Path("datasets/esci/data_vectors")


def register_trace_requests(name):
    """Record the request files of a trace workload (name, size, SHA-256) in the run's result."""
    try:
        from recal.measure.jobs_util import register_data_file
    except ImportError:
        return
    from simulations.approximate_hits.run_e2e_oracle_trace import AKER, WORKLOADS
    for f in WORKLOADS[name]["trace"]:
        register_data_file(AKER / f)


def load_prep(prep, native=False):
    """(db, q, top, kind): db in the index's stored type if native (int8 for SPACEV) else float32
    (memory-mapped where possible), queries float32, exact top-500 ids (memory-mapped)."""
    prep = Path(prep)
    meta_path = prep / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"kind": "esci", "n_db": 1000000}
    top = np.load(prep / "top500.npy", mmap_mode="r")
    if meta.get("kind") == "trace":
        from simulations.approximate_hits.run_e2e_oracle_trace import load_base, load_base_f32
        register_trace_requests(meta["trace"])
        db = load_base(meta["trace"]) if native else load_base_f32(meta["trace"])
        q = np.ascontiguousarray(np.load(prep / "q.npy"), dtype=np.float32)
        return db, q, top, "trace"
    db = np.load(ESCI / f"esci_db{meta.get('n_db', 1000000)}_s42.npy", mmap_mode="r")
    q = np.ascontiguousarray(np.load(ESCI / "esci_queries.npy"), dtype=np.float32)
    return db, q, top, "esci"
