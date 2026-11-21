#!/usr/bin/env python3
"""ReCal as a method of the experiment harness: one cell over a prepared workload.

Reads a prep folder of the advisor's pipeline (results/online/<prep>: meta.json, top500.npy, streams.npz, q.npy),
runs ReCal over a stream with a backend, and writes <out>/recal<tag>.json in the shape of online_sim's results
(method, args, log per batch, events, cost) plus bytes by structure, background work by kind and, with
--timing, the recorded decisions and cache-path times per request (for the throughput replay).

  PYTHONPATH=. .venv-core/bin/python -m recal.sim --prep simulations/approximate_hits/results/online/esci50k \\
      --stream random --budget 0.05 --out <dir>
"""

import json
import time
import shutil
import argparse
import tempfile
from pathlib import Path

import numpy as np

from recal.backend import ExactBackend
from recal.measure.jobs_util import stamp_result
from recal.params import Params
from recal.stored import StoredAnswers
from recal.system import ReCal
from recal import accounting


def load_prep(prep: Path):
    meta = json.loads((prep / "meta.json").read_text()) if (prep / "meta.json").exists() else {"kind": "esci"}
    if meta["kind"] == "files":          # database and queries as .npy files (local checks, TripClick subsample)
        db = np.load(meta["db"])
        q = np.load(meta["q"], mmap_mode="r")
    elif meta["kind"] == "esci":
        db = np.load(f"datasets/esci/data_vectors/esci_db{meta['n_db']}_s42.npy")
        q = np.load("datasets/esci/data_vectors/esci_queries.npy", mmap_mode="r")
    else:
        from simulations.approximate_hits.run_e2e_oracle_trace import load_base_f32
        db = load_base_f32(meta["trace"])
        q = np.load(prep / "q.npy", mmap_mode="r")
    top = np.load(prep / "top500.npy", mmap_mode="r") if (prep / "top500.npy").exists() else None
    return db, q, top, np.load(prep / "streams.npz")


def make_backend(kind: str, db, q, top, stream_idx, args):
    if kind == "exact":
        return ExactBackend(db)
    if top is None:
        raise SystemExit("this backend needs top500.npy in the prep folder")
    if kind == "stored":
        return StoredAnswers(db, q[np.unique(stream_idx)], top[np.unique(stream_idx)])
    z = np.load(args.backend_file)     # DiskANN answers: answer_<target> (K) and deep_<target> (deeper lists)
    u = np.unique(stream_idx)
    return StoredAnswers(db, q[u], z[f"deep_{args.backend_target}"][u], shallow=z[f"answer_{args.backend_target}"][u])


def run(args) -> dict:
    prep = Path(args.prep)
    db, q, top, streams = load_prep(prep)
    idx = np.asarray(streams[args.stream], dtype=np.int64)
    if args.max_requests:
        idx = idx[:args.max_requests]
    d, K = db.shape[1], args.k
    backend = make_backend(args.backend, db, q, top, idx, args)
    M = accounting.budget_bytes(len(db), d, args.budget)
    params = Params(K=K, seed=args.seed, split_by_query=args.split_by_query, idealized=args.idealized,
                    cold_start=args.cold_start, H=args.history, min_ess=args.min_ess)
    work = tempfile.mkdtemp(prefix="recal_sim_")
    system = ReCal(backend, params, M, work)
    log, ev_pos, audit_pos, timing = [], [], [], {"request_ids": [], "hit": [], "dt_ms": []}
    batch = {"n": 0, "served": 0, "ret": 0.0, "audited": 0}
    first_construction, t0 = None, time.time()
    for pos, qi in enumerate(idx):
        vec = np.asarray(q[qi], dtype=np.float32)
        r = system.serve(vec, K)
        served = r.source in ("mini", "main")
        batch["n"] += 1
        if served:
            batch["served"] += 1
            truth = top[qi, :K] if top is not None else backend.search(vec, K, kind="eval")[0]
            batch["ret"] += float(np.isin(truth, r.ids).mean())
            if args.log_served:
                ev_pos.append(pos)
        if r.audited:
            batch["audited"] += 1
            audit_pos.append(pos)
        if first_construction is None and system.constructions:
            first_construction = pos
        if args.timing:
            timing["request_ids"].append(pos)
            timing["hit"].append(bool(served))
            timing["dt_ms"].append(r.cache_ms)
        if batch["n"] == args.batch:
            b = system.bytes_by_structure()
            log.append({"t": pos + 1, "n": batch["n"], "served": batch["served"], "served_retention_sum": batch["ret"],
                        "audited_served": batch["audited"], "lambda": system.lam, "memory": b["total"] / (4 * d),
                        "cached": system.core.n_data_slots(), "entries": system.core.n_entries(),
                        "events": len(system.signals.events), "mini": system.core.mini_size()})
            batch = {"n": 0, "served": 0, "ret": 0.0, "audited": 0}
    bytes_ = system.bytes_by_structure()
    out = {
        "method": "recal", "args": dict(vars(args), warmup=first_construction or 0), "log": log,
        "events": system.signals.events, "constructions": system.constructions,
        "cost": {"topk": backend.stats.get("miss", {}).get("searches", 0) + backend.stats.get("audit", {}).get("searches", 0),
                 "topR": backend.stats.get("candidate", {}).get("searches", 0)},
        "background": backend.stats, "counts": system.counts, "status": system.status_report(),
        "bytes": {k: float(v) for k, v in bytes_.items()},
        "bytes_first_construction": system.first_construction_bytes, "M": M, "within_M": bool(bytes_["total"] <= M),
        "info": [getattr(system.built, "info", None)] if system.built else [], "seconds": time.time() - t0,
    }
    if args.log_served:
        out["served_pos"], out["audit_pos"] = ev_pos, audit_pos
    if args.timing:
        out["timing"] = dict(timing, every=1, n_timed=len(timing["hit"]))
    shutil.rmtree(work, ignore_errors=True)
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--stream", default="random")
    ap.add_argument("--budget", type=float, default=0.05, help="M as a fraction of the dataset's stored bytes")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--backend", choices=["exact", "stored", "file"], default="stored")
    ap.add_argument("--backend-file", default=None)
    ap.add_argument("--backend-target", default="0.95")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold-start", type=int, default=10_000)
    ap.add_argument("--history", type=int, default=30_000, help="H, main-path requests")
    ap.add_argument("--min-ess", type=float, default=100.0)
    ap.add_argument("--split-by-query", action="store_true", help="roles by distinct query (default: by position)")
    ap.add_argument("--idealized", action="store_true", help="exact labels, candidates from all entries (no m cap)")
    ap.add_argument("--timing", action="store_true", help="record hit and cache-path time per request")
    ap.add_argument("--log-served", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    return ap


def main() -> None:
    args = parser().parse_args()
    result = stamp_result(run(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".recal{args.tag}.json.tmp"
    tmp.write_text(json.dumps(result, default=str))
    tmp.replace(out / f"recal{args.tag}.json")


if __name__ == "__main__":
    main()
