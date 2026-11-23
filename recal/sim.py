#!/usr/bin/env python3
"""ReCal as a method of the experiment harness: one cell over a prepared workload.

Reads a prep folder written by online_prepare.py (results/online/<prep>: meta.json, top500.npy, streams.npz, q.npy),
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
from recal.measure.jobs_util import add_hash_flag, register_args, stamp_result
from recal.params import Params
from recal.records import Records, add_response
from recal.stored import StoredAnswers
from recal.system import ReCal
from recal import accounting, history_mode, ksets
from recal.update_schedule import UpdateStream, live_truth


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


ABLATIONS = ("no_union", "no_expansion", "no_shells", "lru", "no_calibration", "no_recalibration",
             "no_mini_cache", "no_rebuild_signals")


def ablations(args) -> dict:
    return {name: bool(getattr(args, name, False)) for name in ABLATIONS}


def run(args, records: Records | None = None) -> dict:
    prep = Path(args.prep)
    db, q, top, streams = load_prep(prep)
    idx = np.asarray(streams[args.stream], dtype=np.int64)
    if args.max_requests:
        idx = idx[:args.max_requests]
    d, K = db.shape[1], args.k
    history_len = history_mode.history_length(len(idx), args.history_frac)
    ks = ksets.k_sequence(len(idx), K, args.k_mixed, args.k_set, args.seed)
    updates = None
    if args.updates:
        if args.backend != "exact":
            raise SystemExit("--updates needs --backend exact (or the live pgvector backend)")
        updates = UpdateStream(dict(np.load(args.updates)))
        backend = ExactBackend(db[updates.present], np.flatnonzero(updates.present))
    else:
        backend = make_backend(args.backend, db, q, top, idx, args)
    M = accounting.budget_bytes(len(db), d, args.budget, args.coord_bytes)
    params = Params(K=K, coord_bytes=args.coord_bytes, seed=args.seed, split_by_query=args.split_by_query, idealized=args.idealized,
                    cold_start=args.cold_start, H=args.history, min_ess=args.min_ess,
                    **ablations(args))
    work = tempfile.mkdtemp(prefix="recal_sim_")
    system = ReCal(backend, params, M, work)
    if records is not None:
        system.enable_records()
    log, ev_pos, audit_pos, timing = [], [], [], {"request_ids": [], "hit": [], "dt_ms": [], "rereads": []}
    batch = {"n": 0, "served": 0, "ret": 0.0, "audited": 0}
    first_construction, t0 = None, time.time()
    if history_len:
        system.ingest_history(np.asarray(q[idx[:history_len]], dtype=np.float32),
                              ks[:history_len])
        first_construction = history_len
    for pos in range(history_len, len(idx)):
        qi = idx[pos]
        vec = np.asarray(q[qi], dtype=np.float32)
        if updates is not None:
            ins, dels = updates.due(pos)
            if len(ins):
                system.insert(ins, db[ins])
            if len(dels):
                system.delete(dels)
        Kp = int(ks[pos])
        r = system.serve(vec, Kp)
        served = r.source in ("mini", "main")
        batch["n"] += 1
        truth = None
        if served or records is not None:
            if updates is not None:
                truth = live_truth(np.asarray(top[qi]), updates.present, Kp)
            else:
                truth = top[qi, :Kp] if top is not None else backend.search(vec, Kp, kind="eval")[0]
        if records is not None:
            add_response(records, pos, Kp, r, truth)
        if served:
            batch["served"] += 1
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
            timing["rereads"].append(r.rereads)
        if batch["n"] == args.batch:
            b = system.bytes_by_structure()
            log.append({"t": pos + 1, "n": batch["n"], "served": batch["served"], "served_retention_sum": batch["ret"],
                        "audited_served": batch["audited"], "lambda": system.lam, "memory": b["total"] / (4 * d),
                        "cached": system.n_cached(), "entries": system.n_entries(),
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
    history_mode.stamp(out, len(idx), history_len)
    if records is not None:
        records.constructions = system.construction_log
        records.meta.update(method="recal", K=K, history_len=history_len, M=M,
                            T=params.T, n_scored=len(idx) - history_len)
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
    ap.add_argument("--k-mixed", action="store_true", help="draw K per request from --k-set")
    ap.add_argument("--k-set", type=int, nargs="+", default=list(ksets.K_SET))
    ap.add_argument("--updates", default=None, help="update schedule npz (recal/update_schedule.py): inserts and "
                    "deletes applied while the stream is replayed; truth is filtered to the live vectors")
    ap.add_argument("--coord-bytes", type=float, default=4.0,
                    help="bytes per coordinate of the stored vectors (SPACEV: 1); M and the cache charge use it")
    ap.add_argument("--backend", choices=["exact", "stored", "file"], default="stored")
    ap.add_argument("--backend-file", default=None)
    ap.add_argument("--backend-target", default="0.95")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold-start", type=int, default=10_000)
    ap.add_argument("--history", type=int, default=30_000, help="H, main-path requests")
    ap.add_argument("--history-frac", type=float, default=0.0,
                    help="first fraction of the stream is history; only the rest is scored")
    ap.add_argument("--min-ess", type=float, default=100.0)
    ap.add_argument("--split-by-query", action="store_true", help="roles by distinct query (default: by position)")
    ap.add_argument("--idealized", action="store_true", help="exact labels, candidates from all entries (no m cap)")
    for name in ABLATIONS:
        ap.add_argument("--" + name.replace("_", "-"), dest=name, action="store_true",
                        help="ReCal ablation (see recal/README.md)")
    ap.add_argument("--timing", action="store_true", help="record hit and cache-path time per request")
    ap.add_argument("--log-served", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    add_hash_flag(ap)
    return ap


def main() -> None:
    args = parser().parse_args()
    register_args(args)
    records = Records()
    result = stamp_result(run(args, records))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records.save(out / f"recal{args.tag}_records.npz")
    tmp = out / f".recal{args.tag}.json.tmp"
    tmp.write_text(json.dumps(result, default=str))
    tmp.replace(out / f"recal{args.tag}.json")


if __name__ == "__main__":
    main()
