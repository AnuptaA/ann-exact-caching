#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Real ANN indexes as backends (approximate backend, step 2; context doc 5ab).
Run with .venv-ann (faiss-cpu 1.8.0, diskannpy 0.7.0, numpy 1.25; kept apart
from .venv so its numpy is not downgraded).

For one dataset (fixed database) and one index type:
  1. build the index over the whole database;
  2. tune its search effort on a sample of queries to the smallest setting
     whose mean recall@k (share of the exact top-k found in the returned
     top-k) reaches each --targets value;
  3. with each tuned setting, answer every query twice, as the system would
     ask:
       answer rows  top-k answers (misses, audits): search at the tuned
                    effort (--answer-depth ids kept, >= k)
       deep rows    top-R lists for building cache entries and shells:
                    asking for R results forces a larger search (HNSW /
                    DiskANN need a search list of at least R; IVF keeps its
                    tuned number of probed lists)
  4. measure recall@k and recall@R against the exact ranking, and how
     concentrated the misses are (are the same vectors missed by many
     queries, unlike independent random misses?).
Indexes: hnsw (FAISS, in memory; M 32, efConstruction 200), ivf (FAISS
IVF-Flat, nlist 4096; the algorithm of pgvector's IVFFlat), diskann
(diskannpy disk index; R 64, L 100, in-memory compressed vectors sized by
--diskann-pq-bytes per vector). Writes results/backends/<prep>/<index>.npz
(answer / deep rows per target, measured recall, tuned setting) and .json.
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RES = Path("simulations/approximate_hits/results")
ESCI = Path("datasets/esci/data_vectors")
GRIDS = {"hnsw": [10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 200, 256, 320, 400,
                  512, 640, 800, 1000],
         "ivf": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
         "diskann": [10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 200, 256, 320, 400,
                     512, 640, 800, 1000]}

#-------------------------------------------------------------------------------


def load(prep):
    meta = json.loads((prep / "meta.json").read_text()) if (prep / "meta.json").exists() \
        else {"kind": "esci", "n_db": 1000000}
    if meta["kind"] == "trace":
        # SPACEV: 8-bit integer vectors kept as they are (DiskANN builds an
        # 8-bit index; queries are integer-valued); truth = exact top-500
        from simulations.approximate_hits.run_e2e_oracle_trace import load_base
        db = load_base(meta["trace"])
        q = np.ascontiguousarray(np.load(prep / "q.npy"), dtype=np.float32)
        truth = np.load(prep / "top500.npy")
        return db, q, truth
    db = np.ascontiguousarray(np.load(ESCI / f"esci_db{meta['n_db']}_s42.npy"), dtype=np.float32)
    q = np.ascontiguousarray(np.load(ESCI / "esci_queries.npy"), dtype=np.float32)
    truth = np.load(prep / "topU1000.npy")
    return db, q, truth


def recall(truth, ids, k):
    t, a = truth[:, :k], ids[:, :k]
    return float(np.mean([np.intersect1d(x, y).size / k for x, y in zip(t, a)]))


class Index:
    def __init__(self, kind, db, args, work):
        self.kind, self.threads = kind, args.threads
        self.beam = getattr(args, "beam", 2)
        self.qdtype = db.dtype if db.dtype in (np.int8, np.uint8) else np.float32
        t0 = time.time()
        if kind == "hnsw":
            import faiss
            self.ix = faiss.IndexHNSWFlat(db.shape[1], 32)
            self.ix.hnsw.efConstruction = 200
            self.ix.add(db)
        elif kind == "ivf":
            import faiss
            quant = faiss.IndexFlatL2(db.shape[1])
            self.ix = faiss.IndexIVFFlat(quant, db.shape[1], 4096)
            rng = np.random.default_rng(0)
            self.ix.train(db[rng.choice(len(db), 200000, replace=False)])
            self.ix.add(db)
        else:
            import diskannpy
            d = Path(args.index_dir) if getattr(args, "index_dir", None) else work / "diskann"
            if not (d / "ann_disk.index").exists():
                d.mkdir(parents=True, exist_ok=True)
                gb = args.diskann_pq_bytes * len(db) / 1e9
                diskannpy.build_disk_index(
                    data=db, distance_metric="l2", index_directory=str(d), complexity=100,
                    graph_degree=64, search_memory_maximum=gb, build_memory_maximum=args.build_gb,
                    num_threads=args.threads, pq_disk_bytes=0, index_prefix="ann")
            from simulations.approximate_hits.diskann_aio import open_disk_index
            self.ix = open_disk_index(
                index_directory=str(d), num_threads=args.threads, num_nodes_to_cache=0,
                index_prefix="ann")
        self.build_s = time.time() - t0
        logger.info(f"{kind}: built in {self.build_s:.0f} s")

    def search(self, q, k, effort):
        if self.kind == "hnsw":
            self.ix.hnsw.efSearch = max(effort, k)
            return self.ix.search(q, k)[1]
        if self.kind == "ivf":
            self.ix.nprobe = effort
            return self.ix.search(q, k)[1]
        q = q.astype(self.qdtype)
        ids, _ = self.ix.batch_search(q, k_neighbors=k, complexity=max(effort, k),
                                      beam_width=self.beam, num_threads=self.threads)
        return ids.astype(np.int64)


def miss_concentration(truth, ids, k):
    """How concentrated misses are on particular vectors: share of all
    misses (true top-k vectors not returned) that fall on the 10% of
    (query-relevant) vectors missed most often, vs what independent misses
    at the same rate would give."""
    t = truth[:, :k]
    hit = np.array([np.isin(x, y) for x, y in zip(t, ids[:, :k])])
    flat, miss = t.ravel(), ~hit.ravel()
    n_vec = np.bincount(flat)
    m_vec = np.bincount(flat[miss], minlength=len(n_vec))
    seen = np.flatnonzero(n_vec >= 3)              # vectors relevant to >= 3 queries
    rate = m_vec[seen] / n_vec[seen]
    top = np.argsort(-rate)[:max(1, len(seen) // 10)]
    share = m_vec[seen][top].sum() / max(m_vec[seen].sum(), 1)
    rng = np.random.default_rng(0)
    sim = rng.binomial(n_vec[seen], miss.mean())
    top_s = np.argsort(-(sim / n_vec[seen]))[:len(top)]
    share_ind = sim[top_s].sum() / max(sim.sum(), 1)
    return {"share_of_misses_on_top10pct_vectors": float(share),
            "same_if_independent": float(share_ind), "n_vectors": int(len(seen))}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prep", type=str, default=str(RES / "online" / "esci1m"))
    ap.add_argument("--index", choices=list(GRIDS), required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.99, 0.95, 0.9])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--answer-depth", type=int, default=20)
    ap.add_argument("--tune-queries", type=int, default=5000)
    ap.add_argument("--diskann-pq-bytes", type=int, default=32)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--build-gb", type=float, default=64.0, help="DiskANN build memory cap")
    ap.add_argument("--out-name", type=str, default=None,
                    help="results/backends/<name> (default: the --prep folder's name; "
                         "SPACEV traces share one database: spacev10m)")
    ap.add_argument("--build-only", action="store_true",
                    help="build the index and report recall at --build-effort, no tuning")
    ap.add_argument("--build-effort", type=int, default=128)
    ap.add_argument("--beam", type=int, default=2, help="DiskANN beam width")
    ap.add_argument("--index-dir", type=str, default=None, help="DiskANN index directory (default <out>/diskann)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where the answer file and logs go (default results/backends/<out-name or prep name> under "
                         "simulations/approximate_hits)")
    ap.add_argument("--fixed", nargs="+", default=None,
                    help="target:search_list pairs chosen by select_backend.py (no tuning curve): answers and "
                         "deep lists at exactly that search list and --beam (doc 9u)")
    ap.add_argument("--efforts", type=int, nargs="+", default=None,
                    help="search efforts to try (default: the index's grid, GRIDS)")
    ap.add_argument("--share-deep", action="store_true",
                    help="--fixed with search list >= R: one search per query for answer and deep rows (doc 9af)")
    args = ap.parse_args()
    prep = Path(args.prep)
    out = Path(args.out_dir) if args.out_dir else RES / "backends" / (args.out_name or prep.name)
    out.mkdir(parents=True, exist_ok=True)
    db, q, truth = load(prep)
    ix = Index(args.index, db, args, out)
    rng = np.random.default_rng(1)
    tune = rng.choice(len(q), args.tune_queries, replace=False)
    if args.build_only:
        r = recall(truth[tune], ix.search(q[tune], args.k, args.build_effort), args.k)
        info = {"index": args.index, "build_s": ix.build_s, "pq_bytes": args.diskann_pq_bytes,
                "effort": args.build_effort, "recall_at_k": r, "n_db": len(db)}
        (out / f"{args.index}_build.json").write_text(json.dumps(info, indent=1))
        logger.info(f"built: {info}")
        return
    if args.fixed:          # chosen settings: no curve; answers at exactly these search lists
        res = {"index": args.index, "index_dir": str(args.index_dir or out / "diskann"), "beam": args.beam,
               "build_s": ix.build_s, "curve": {}, "targets": {}}
        arrays = {}
        for pair in args.fixed:
            tgt, e = pair.split(":")
            e = int(e)
            if args.share_deep and e >= args.R:
                # one search: at search list e >= R the top-R list's first answer_depth ids are the
                # top-answer_depth search's (k only cuts the output); checked on 200 queries first
                chk = rng.choice(len(q), 200, replace=False)
                same = np.mean([np.array_equal(a, b) for a, b in
                                zip(ix.search(q[chk], args.answer_depth, e), ix.search(q[chk], args.R, e)[:, :args.answer_depth])])
                logger.info(f"share-deep check: {same:.3f} of 200 queries identical")
                if same < 0.99:
                    raise SystemExit("share-deep: answers differ, run without --share-deep")
                deep = ix.search(q, args.R, e)
                ans = np.ascontiguousarray(deep[:, :args.answer_depth])
            else:
                ans = ix.search(q, args.answer_depth, e)
                deep = ix.search(q, args.R, e)
            stats = {"effort": e, "beam": args.beam, "recall_answer_at_k": recall(truth, ans, args.k),
                     "recall_deep_at_k": recall(truth, deep, args.k),
                     "recall_deep_at_R": recall(truth, deep, args.R)}
            res["targets"][tgt] = stats
            arrays[f"answer_{tgt}"] = ans.astype(np.int32)
            arrays[f"deep_{tgt}"] = deep.astype(np.int32)
            logger.info(f"target {tgt}: {stats}")
        np.savez(out / f"{args.index}.npz", **arrays)
        (out / f"{args.index}.json").write_text(json.dumps(res, indent=1))
        logger.info(f"wrote {out / args.index}.npz")
        return
    curve = {}
    for e in args.efforts or GRIDS[args.index]:
        t0 = time.time()
        r = recall(truth[tune], ix.search(q[tune], args.k, e), args.k)
        curve[e] = (r, (time.time() - t0) / len(tune) * 1e3)
        logger.info(f"{args.index} effort {e}: recall@{args.k} {r:.4f} "
                    f"({curve[e][1]:.3f} ms / query, batched)")
        if r >= max(args.targets) + 0.005:
            break
    res = {"index": args.index, "build_s": ix.build_s, "curve": curve, "targets": {}}
    arrays = {}
    for tgt in args.targets:
        ok = [e for e, (r, _) in curve.items() if r >= tgt]
        if not ok:
            logger.warning(f"target {tgt} not reached")
            continue
        e = min(ok)
        ans = ix.search(q, args.answer_depth, e)
        deep = ix.search(q, args.R, e)
        stats = {"effort": e, "recall_answer_at_k": recall(truth, ans, args.k),
                 "recall_deep_at_k": recall(truth, deep, args.k),
                 "recall_deep_at_R": recall(truth, deep, args.R),
                 "misses": miss_concentration(truth, ans, args.k)}
        res["targets"][str(tgt)] = stats
        arrays[f"answer_{tgt}"] = ans.astype(np.int32)
        arrays[f"deep_{tgt}"] = deep.astype(np.int32)
        logger.info(f"target {tgt}: {stats}")
    np.savez(out / f"{args.index}.npz", **arrays)
    (out / f"{args.index}.json").write_text(json.dumps(res, indent=1))
    logger.info(f"wrote {out / args.index}.npz")


if __name__ == "__main__":
    main()
