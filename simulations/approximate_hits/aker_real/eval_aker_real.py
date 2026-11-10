#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Run the REAL Aker (github.com/sjoon-oh/aker, libaker.so built unmodified, via
aker_shim.cc) on our workloads. Run with .venv-ann and the LD_LIBRARY_PATH
printed by build.sh.

Request loop, as in the authors' integration: for each request of the trace,
look the query up in Aker; on an exact or approximate hit the answer is the
first k cached neighbors (an approximate hit also links the query to its
representative); on a miss the backend answers, and the answer is inserted.
Requests are processed one at a time (contents change immediately, unlike our
batched reimplementation in online_baselines.py).

Modes
  quality  the backend's answers are precomputed (exact top-500 of the prep
           folder, or --answers FILE: DiskANN answers); no backend time.
           Logs per 1000-request batch, in the simulator's format (t, n, served,
           served_retention_sum, ...), for the requests after --warmup (the
           first --warmup requests fill the cache and are not logged).
  timing   (--timing, run_timing) real DiskANN on misses.

Hit kinds: exact (the query's own entry), approximate (a representative within
its threshold; the query is then linked to it), linked (a repeat of a query
that was served approximately before: Aker returns the representative's answer).

Retention of a hit = share of the exact top-k found among the k neighbors Aker
returns; also reported against the backend's own answer for the query.
Budget: pool_size = budget x database size vector slots (Aker counts raw
neighbor vectors; reserve neighbors (top_delta) occupy slots too).
"""

import sys
import json
import time
import ctypes
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
RES = Path("simulations/approximate_hits/results")
AKER = Path("datasets/aker")
CONFIG = Path.home() / "third_party" / "aker" / "bootstrap" / "aker-standard.ini"

#-------------------------------------------------------------------------------


class Aker:
    """ctypes wrapper of aker_shim.cc."""

    def __init__(self, dim, pool, topk, delta, risk, a_tighten, a_loosen):
        lib = ctypes.CDLL(str(HERE / "libakershim.so"))
        P = ctypes.c_void_p
        lib.shim_create.restype = P
        lib.shim_create.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32,
                                    ctypes.c_uint32, ctypes.c_float, ctypes.c_float,
                                    ctypes.c_float, ctypes.c_char_p]
        lib.shim_lookup.restype = ctypes.c_int
        lib.shim_lookup.argtypes = [P, ctypes.c_uint64, P, P, P]
        lib.shim_insert.restype = ctypes.c_int
        lib.shim_insert.argtypes = [P, ctypes.c_uint64, P, ctypes.c_uint32, P, P, P]
        lib.shim_write.restype = ctypes.c_double
        lib.shim_write.argtypes = [P, ctypes.c_uint64, P]
        lib.shim_process.restype = ctypes.c_double
        lib.shim_process.argtypes = [P]
        lib.shim_delete.restype = ctypes.c_double
        lib.shim_delete.argtypes = [P, ctypes.c_uint64]
        lib.shim_status.restype = ctypes.c_char_p
        lib.shim_status.argtypes = [P]
        lib.shim_destroy.argtypes = [P]
        self.lib, self.dim, self.topk, self.delta = lib, dim, topk, delta
        self.h = lib.shim_create(dim, pool, topk, delta, risk, a_tighten, a_loosen,
                                 str(CONFIG).encode())
        if not self.h:
            raise SystemExit("shim_create failed")
        self.ids = np.zeros(topk, dtype=np.int64)
        self.t = np.zeros(2, dtype=np.float64)
        self.ti = ctypes.c_double(0.0)

    def lookup(self, qid, q):
        """kind (0 miss, 1 exact hit, 2 approximate hit, 3 invalid, 4 linked hit), ids, seconds."""
        kind = self.lib.shim_lookup(self.h, qid, q.ctypes.data, self.ids.ctypes.data,
                                    self.t.ctypes.data)
        return kind, self.ids, self.t[0], self.t[1]

    def insert(self, qid, q, nid, nvec):
        ok = self.lib.shim_insert(self.h, qid, q.ctypes.data, len(nid), nid.ctypes.data,
                                  nvec.ctypes.data, ctypes.byref(self.ti))
        return ok, self.ti.value

    def write(self, vid, v):
        return self.lib.shim_write(self.h, vid, v.ctypes.data)

    def process(self):
        return self.lib.shim_process(self.h)

    def delete(self, vid):
        return self.lib.shim_delete(self.h, vid)

    def status(self):
        return self.lib.shim_status(self.h).decode(errors="replace")

    def close(self):
        """Destroy the cache. Aker then exports telemetry to /tmp/aker_trace_<time>_pid<PID>_gen<G>/
        (~300 MB per run, and /tmp is RAM-backed here); remove it, as the harness keeps
        its own summary and Aker's status text."""
        import os
        import glob
        import shutil
        self.lib.shim_destroy(self.h)
        for d in glob.glob(f"/tmp/aker_trace_*_pid{os.getpid()}_*"):
            shutil.rmtree(d, ignore_errors=True)


def load(prep, stream_name="trace", native=False):
    """native: the database in its stored type (int8 for SPACEV, as the DiskANN
    index takes it); else float32 (a shared memory map when available)."""
    meta = json.loads((prep / "meta.json").read_text())
    from simulations.approximate_hits.run_e2e_oracle_trace import load_base, load_base_f32
    db = load_base(meta["trace"]) if native else load_base_f32(meta["trace"])
    q = np.ascontiguousarray(np.load(prep / "q.npy"), dtype=np.float32)
    top = np.load(prep / "top500.npy", mmap_mode="r")
    stream = np.load(prep / "streams.npz")[stream_name]
    return db, q, top, stream


class World:
    """The database and the backend, as in online_sim.py's load():
    truth(qi, n) = first n present ids of the exact ranking; answer(qi, n) =
    the backend's (exact, or with --backend-recall < 1 the simulator's
    approximate_rows model: each vector missed independently per query);
    with --data-change the database starts at the initial presence mask and
    events(i) returns the inserts / deletes scheduled at or before request i
    (applied before it, one request at a time)."""

    def __init__(self, args, prep, top):
        from simulations.approximate_hits.online_sim import approximate_rows, measured_recall
        self.present, self.ev = None, None
        kind = args.data_change
        if kind != "none":
            topU = np.load(prep / "topU1000.npy")
            extra = np.load(prep / f"rank_extra_{kind}.npz")
            width = extra["rows"].shape[1] if len(extra["flagged"]) else topU.shape[1]
            rank = np.full((len(topU), width), -1, dtype=np.int32)
            rank[:, :topU.shape[1]] = topU
            if len(extra["flagged"]):
                rank[extra["flagged"]] = extra["rows"]
            dc = np.load(prep / f"data_{kind}.npz")
            n = len(dc["ins_ids"])
            if args.schedule == "burst":
                t_ins = t_del = np.full(n, int(dc["burst_at"]))
            else:
                t_ins, t_del = dc["ins_t"], dc["del_t"]
            t = np.r_[t_ins, t_del]
            order = np.argsort(t, kind="stable")
            self.ev = {"t": t[order], "id": np.r_[dc["ins_ids"], dc["del_ids"]][order],
                       "insert": np.r_[np.ones(n, bool), np.zeros(n, bool)][order]}
            self.ev_i = 0
            self.present = dc["initial"].copy()
            self.truth_rows = rank
        elif args.backend_recall < 1.0:
            self.truth_rows = np.load(prep / "topU1000.npy")
        else:
            self.truth_rows = top
        if args.backend_file:
            # a real index (eval_real_backend.py): its top-20 answers at the
            # search effort tuned to --backend-target recall@10
            if kind != "none":
                raise SystemExit("real-index backends support a fixed database only")
            z = np.load(args.backend_file)
            self.answer_rows = z[f"answer_{args.backend_target}"]
            if self.answer_rows.shape[1] < args.k + args.delta:
                raise SystemExit("answer rows shallower than k + delta")
            self.recall_measured = measured_recall(np.asarray(self.truth_rows),
                                                   self.answer_rows, args.k)
        elif args.backend_recall < 1.0:
            self.answer_rows = approximate_rows(np.asarray(self.truth_rows), args.backend_recall,
                                                args.backend_seed)
            self.recall_measured = measured_recall(np.asarray(self.truth_rows),
                                                   self.answer_rows, args.k)
        else:
            self.answer_rows = self.truth_rows
            self.recall_measured = 1.0

    def _first(self, rows, qi, n):
        r = rows[qi]
        m = r >= 0
        if self.present is not None:
            m &= self.present[np.maximum(r, 0)]
        ids = r[m][:n]
        if len(ids) < n:            # pad (rare: fewer than n present / returned)
            ids = np.r_[ids, np.full(n - len(ids), ids[-1] if len(ids) else r[0])]
        return np.asarray(ids, dtype=np.int64)

    def truth(self, qi, n):
        return self._first(self.truth_rows, qi, n)

    def answer(self, qi, n):
        return self._first(self.answer_rows, qi, n)

    def events(self, i):
        """(inserted ids, deleted ids) scheduled at or before request i, not yet applied."""
        if self.ev is None:
            return None, None
        j = int(np.searchsorted(self.ev["t"], i, side="right"))
        if j <= self.ev_i:
            return None, None
        sl = slice(self.ev_i, j)
        ins, ids = self.ev["insert"][sl], self.ev["id"][sl]
        self.ev_i = j
        self.present[ids[ins]] = True
        self.present[ids[~ins]] = False
        return ids[ins], ids[~ins]


def run_quality(args, db, q, world, stream):
    """Replay the stream against World (truth, backend answers, data changes)."""
    k, delta = args.k, args.delta
    n_neigh = k + delta
    pool = int(args.budget * len(db))
    ak = Aker(q.shape[1], pool, k, delta, args.risk, args.alpha_tighten, args.alpha_loosen)
    changing = world.ev is not None
    KIND = {1: "exact", 2: "approx", 4: "linked"}

    def fresh():
        c = {"n": 0, "served": 0, "ret": 0.0, "ret_b": 0.0, "deleted_returned": 0,
             "events": 0}
        for name in KIND.values():
            c[name], c["ret_" + name] = 0, 0.0
        return c

    def flush(t):
        log.append({"t": t, "n": cur["n"], "served": cur["served"],
                    "served_retention_sum": cur["ret"],
                    "served_retention_backend_sum": cur["ret_b"],
                    "deleted_returned": cur["deleted_returned"], "events": cur["events"],
                    **{f"hits_{nm}": cur[nm] for nm in KIND.values()},
                    **{f"retention_sum_{nm}": cur["ret_" + nm] for nm in KIND.values()}})

    log, cur = [], fresh()
    lat = {"lookup_hit": [], "lookup_miss": [], "link": [], "insert": [], "write": [],
           "delete": [], "process": []}
    stats = {"miss": 0, "hit_exact": 0, "hit_approx": 0, "hit_linked": 0, "invalid": 0,
             "insert_failed": 0}
    n_req = len(stream) if args.max_requests is None else min(args.max_requests, len(stream))
    t_start = time.time()
    for i in range(n_req):
        if i and i % args.progress == 0:
            logger.info(f"{i} requests, {time.time() - t_start:.0f} s, {stats}")
        qi = int(stream[i])
        qv = q[qi]
        measure = i >= args.warmup
        ins, dels = world.events(i)
        if ins is not None:
            # database changes reach Aker as they happen: inserts through its
            # write log (fast path), deletes through markVectorDeleted
            for vid in ins:
                tw = ak.write(int(vid) + 1, np.ascontiguousarray(db[vid], dtype=np.float32))
                if measure:
                    lat["write"].append(tw)
            for vid in dels:
                td = ak.delete(int(vid) + 1)
                if measure:
                    lat["delete"].append(td)
            if measure:
                cur["events"] += len(ins) + len(dels)
        kind, ids, t_look, t_link = ak.lookup(qi + 1, qv)
        if kind in KIND:
            if measure:
                got = ids - 1
                r = np.isin(got, world.truth(qi, k)).mean()
                cur["served"] += 1
                cur["ret"] += r
                cur["ret_b"] += np.isin(got, world.answer(qi, k)).mean()
                if changing:
                    cur["deleted_returned"] += int((~world.present[got]).sum())
                cur[KIND[kind]] += 1
                cur["ret_" + KIND[kind]] += r
                lat["lookup_hit"].append(t_look)
                if kind == 2:
                    lat["link"].append(t_link)
            stats["hit_" + KIND[kind]] += 1
        else:
            stats["miss" if kind == 0 else "invalid"] += 1
            nid = world.answer(qi, n_neigh)
            nvec = np.ascontiguousarray(db[nid], dtype=np.float32)
            ok, t_ins = ak.insert(qi + 1, qv, nid + 1, nvec)
            stats["insert_failed"] += 0 if ok else 1
            if measure:
                lat["lookup_miss"].append(t_look)
                lat["insert"].append(t_ins)
        if changing:
            # write-log slow path, called after each request as in the authors' integration
            tp = ak.process()
            if measure:
                lat["process"].append(tp)
        if measure:
            cur["n"] += 1
            if cur["n"] == args.batch:
                flush(i + 1)
                cur = fresh()
    if cur["n"]:
        flush(n_req)
    status = ak.status()
    ak.close()
    n = sum(b["n"] for b in log)
    s = sum(b["served"] for b in log)
    summ = {"hit_rate": s / max(n, 1),
            "retention": sum(b["served_retention_sum"] for b in log) / max(s, 1),
            "retention_vs_backend": sum(b["served_retention_backend_sum"] for b in log) / max(s, 1),
            "exact_hit_share": sum(b["hits_exact"] for b in log) / max(n, 1),
            "approx_hit_share": sum(b["hits_approx"] for b in log) / max(n, 1),
            "linked_hit_share": sum(b["hits_linked"] for b in log) / max(n, 1),
            "retention_by_kind": {nm: sum(b[f"retention_sum_{nm}"] for b in log)
                                  / max(sum(b[f"hits_{nm}"] for b in log), 1)
                                  for nm in KIND.values()},
            "deleted_returned": sum(b["deleted_returned"] for b in log),
            "backend_recall_measured": world.recall_measured,
            "queries": n, "seconds": time.time() - t_start,
            "mean_ms": {k_: float(np.mean(v) * 1e3) if v else None for k_, v in lat.items()},
            "stats": stats}
    return {"args": vars(args), "log": log, "summary": summ, "status": status}


def run_timing(args, db, q, top, stream):
    """Warm the cache on the first --warmup requests (backend answers: exact top
    k + delta, untimed), then time the next --n requests end to end, one at a
    time on one thread: Aker lookup; on an approximate hit the alias link; on a
    miss the DiskANN search (k + delta neighbors, list --L), reading the
    neighbors' vectors and inserting the entry. Then the same requests against
    DiskANN alone, without and with Aker's memory as DiskANN node cache."""
    from simulations.approximate_hits.eval_timing import open_diskann, time_backend, summary
    from simulations.approximate_hits.timing_env import make_guard
    guard = make_guard(args)       # refuses unless the machine is quiet (timing_env.py)
    guard.start()
    k, delta = args.k, args.delta
    n_neigh = k + delta
    pool = int(args.budget * len(db))
    ak = Aker(q.shape[1], pool, k, delta, args.risk, args.alpha_tighten, args.alpha_loosen)
    for i in range(args.warmup):
        qi = int(stream[i])
        kind, _, _, _ = ak.lookup(qi + 1, q[qi])
        if kind in (0, 3):
            nid = np.ascontiguousarray(top[qi, :n_neigh], dtype=np.int64)
            ak.insert(qi + 1, q[qi], nid + 1, np.ascontiguousarray(db[nid], dtype=np.float32))
    import re
    mb = float(re.search(r"Total: ([0-9.]+) MB", ak.status()).group(1))
    req = np.asarray(stream[args.warmup:args.warmup + args.n], dtype=np.int64)
    xs = np.ascontiguousarray(q[req])
    xd = np.ascontiguousarray(xs.astype(db.dtype))      # 8-bit index; queries are integers
    ix = open_diskann(args.diskann, 0)
    time_backend(ix, xd[:args.warm], n_neigh, args.L)   # warm the OS page cache / DiskANN
    kinds = np.zeros(len(xs), dtype=np.int64)
    total = np.empty(len(xs))
    look = np.empty(len(xs))
    ret = np.full(len(xs), np.nan)
    for i, (x, xq) in enumerate(zip(xs, xd)):
        qi = int(req[i])
        a = time.perf_counter()
        kind, ids, t_look, _ = ak.lookup(qi + 1, x)
        if kind in (0, 3):
            nid, _ = ix.search(xq, k_neighbors=n_neigh, complexity=args.L, beam_width=2)
            nid = np.ascontiguousarray(nid, dtype=np.int64)
            ak.insert(qi + 1, x, nid + 1, np.ascontiguousarray(db[nid], dtype=np.float32))
        total[i] = time.perf_counter() - a
        if kind not in (0, 3):
            ret[i] = np.isin(ids - 1, top[qi, :k]).mean()      # bookkeeping, not timed
        look[i] = t_look
        kinds[i] = kind
    hit = np.isin(kinds, (1, 2, 4))
    res = {"args": vars(args), "n": len(xs), "aker_mb": mb, "hit_rate": float(hit.mean()),
           "kinds": {nm: float(np.mean(kinds == c)) for c, nm in
                     ((1, "exact"), (2, "approx"), (4, "linked"), (0, "miss"), (3, "invalid"))},
           "retention": float(np.nanmean(ret)) if hit.any() else None,
           "lookup_ms": summary(look),
           "cache": dict(summary(total), hits_ms=summary(total[hit]) if hit.any() else None,
                         misses_ms=summary(total[~hit]) if (~hit).any() else None)}
    logger.info(f"hit rate {res['hit_rate']:.3f} {res['kinds']}; retention {res['retention']}; "
                f"Aker {mb:.1f} MB; Aker + DiskANN {res['cache']}")
    node_bytes = db.shape[1] * db.dtype.itemsize + 4 + 64 * 4   # as eval_timing.py
    n_nodes = int(mb * 1e6) // node_bytes
    for arm, nodes in (("backend", 0), ("backend_mem", n_nodes)):
        if arm == "backend_mem":
            del ix
            ix = open_diskann(args.diskann, nodes)
        time_backend(ix, xd[:args.warm], k, args.L)
        res[arm] = dict(summary(time_backend(ix, xd, k, args.L)), nodes_cached=int(nodes))
        logger.info(f"{arm}: {res[arm]}")
    m = res["backend"]["mean_ms"]
    res["speedup"] = {"overall": m / res["cache"]["mean_ms"],
                      "limit_1_over_1_minus_h": 1 / max(1 - res["hit_rate"], 1e-9),
                      "vs_equal_memory": res["backend_mem"]["mean_ms"] / res["cache"]["mean_ms"]}
    logger.info(f"speedup {res['speedup']}")
    ak.close()
    guard.stop()
    res["environment"] = guard.report()
    logger.info(f"environment: {res['environment']['verdict']} "
                f"(other processes: mean {res['environment']['during']['other_cores_mean']} cores, "
                f"max {res['environment']['during']['other_cores_max']})")
    return res


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prep", type=str, default=str(RES / "online" / "spacev-aker-0.99"))
    ap.add_argument("--budget", type=float, default=0.01, help="pool size, share of the database")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--delta", type=int, default=5, help="Aker top_delta (reserve neighbors)")
    ap.add_argument("--risk", type=float, default=0.35)
    ap.add_argument("--alpha-tighten", type=float, default=0.9)
    ap.add_argument("--alpha-loosen", type=float, default=1.1)
    ap.add_argument("--warmup", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--progress", type=int, default=5000, help="log every this many requests")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--stream", choices=["trace", "random", "shift", "multishift"],
                    default="trace", help="request stream of the prep (streams.npz)")
    ap.add_argument("--data-change", choices=["none", "churn", "drift"], default="none",
                    help="database inserts / deletes (online_prepare_data.py)")
    ap.add_argument("--schedule", choices=["gradual", "burst"], default="gradual")
    ap.add_argument("--backend-recall", type=float, default=1.0,
                    help="simulated approximate backend (online_sim.approximate_rows)")
    ap.add_argument("--backend-seed", type=int, default=123)
    ap.add_argument("--backend-file", type=str, default=None,
                    help="real-index answers (results/backends/<prep>/<index>.npz)")
    ap.add_argument("--backend-target", type=str, default="0.95")
    ap.add_argument("--timing", action="store_true",
                    help="end-to-end timing with real DiskANN on misses (run_timing)")
    ap.add_argument("--n", type=int, default=1000, help="timing: requests timed after warm-up")
    ap.add_argument("--warm", type=int, default=200, help="timing: untimed searches first, per arm")
    ap.add_argument("--L", type=int, default=128, help="timing: DiskANN search list")
    ap.add_argument("--diskann", type=str, default=str(RES / "backends" / "spacev10m" / "diskann"))
    from simulations.approximate_hits.timing_env import add_args
    add_args(ap)
    args = ap.parse_args()
    sys.path.append(str(HERE.parent.parent.parent))
    prep = Path(args.prep)
    db, q, top, stream = load(prep, args.stream, native=args.timing)
    if args.timing:
        res = run_timing(args, db, q, top, stream)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res, indent=1))
        return
    res = run_quality(args, db, q, World(args, prep, top), stream)
    phase_key = f"phase_{args.stream}" if args.stream != "trace" else None
    streams = np.load(prep / "streams.npz")
    if phase_key in streams.files or (args.stream == "shift" and "phase" in streams.files):
        ph = streams[phase_key if phase_key in streams.files else "phase"][:len(stream)]
        res["phase_start"] = int(np.argmax(ph == 1))
        res["phase_starts"] = [0] + (np.flatnonzero(np.diff(ph)) + 1).tolist()

    s = res["summary"]
    logger.info(f"hit rate {s['hit_rate']:.4f} (exact {s['exact_hit_share']:.4f}, approx "
                f"{s['approx_hit_share']:.4f}, linked {s['linked_hit_share']:.4f}), retention "
                f"vs exact {s['retention']:.4f} (by kind {s['retention_by_kind']}), "
                f"vs backend {s['retention_vs_backend']:.4f}; mean ms {s['mean_ms']}; "
                f"{s['seconds']:.0f} s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
