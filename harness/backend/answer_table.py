"""Answer tables: L backend neighbors (ids and distances) for every request vector.

Distances are recomputed in float32 with the shared distance function, never
taken from the backend. Resumable per block; blocks are written atomically.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from workloads.q1.distance import l2_to_ids

logger = logging.getLogger(__name__)

BLOCK = 2000


def build_answer_table(backend, vectors, base, out_dir: Path, list_size: int, ef_search: int,
                       block: int = BLOCK, shard: tuple[int, int] = (0, 1)) -> None:
    """Write blocks of (ids, distances); shard=(i, n) takes every n-th block."""
    if len(base) >= 2 ** 31:
        raise ValueError("answer tables store int32 ids; the base has 2^31 or more rows")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"n": len(vectors), "list_size": list_size, "ef_search": ef_search,
            "block": block, "backend": backend.name}
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError(f"{meta_path} belongs to a different table")
    tmp_meta = meta_path.with_name(f"meta.{os.getpid()}.tmp")   # shards write the same content
    tmp_meta.write_text(json.dumps(meta))
    tmp_meta.replace(meta_path)
    mine = [b for b in range(-(-len(vectors) // block)) if b % shard[1] == shard[0]]
    todo = [b for b in mine if not (out_dir / f"block_{b:07d}.npz").exists()]
    logger.info(f"answer table shard {shard[0]}/{shard[1]}: {len(todo)} of {len(mine)} blocks to build")
    began = time.time()
    for b, start in ((b, b * block) for b in todo):
        path = out_dir / f"block_{b:07d}.npz"
        q = np.asarray(vectors[start:start + block], dtype=np.float32)
        ids = backend.search(q, list_size, ef_search)
        if (ids < 0).any() or (ids >= len(base)).any():
            raise ValueError(f"block {b}: backend returned fewer than {list_size} valid ids "
                             f"(ef_search {ef_search} too small for the list size?)")
        dist = np.stack([l2_to_ids(qi, base, row) for qi, row in zip(q, ids)]).astype(np.float32)
        order = np.argsort(dist, axis=1, kind="stable")   # backend order may differ slightly
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.take_along_axis(ids, order, 1).astype(np.int32),
                 dist=np.take_along_axis(dist, order, 1))
        tmp.rename(path)
        built = todo.index(b) + 1
        per_block = (time.time() - began) / built
        logger.info(f"shard {shard[0]}: block {b} done ({built}/{len(todo)}, {block / per_block:.0f} "
                    f"queries/s, ETA {(len(todo) - built) * per_block / 60:.1f} min)")


def load_answer_table(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    files = sorted(out_dir.glob("block_*.npz"))
    expected = -(-meta["n"] // meta["block"])
    if len(files) != expected:
        raise FileNotFoundError(f"{len(files)} of {expected} blocks present in {out_dir}")
    parts = [np.load(f) for f in files]
    return (np.concatenate([p["ids"] for p in parts]), np.concatenate([p["dist"] for p in parts]))
