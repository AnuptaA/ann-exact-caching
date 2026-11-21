"""Part A step 2: map sphere-10m rows to Hermes SPHERE_100K parquet rows."""

import sys
import time

import numpy as np
import pyarrow.parquet as pq

from common import AKER, cosine_rows, read_npy_rows

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
t0 = time.time()
pf = pq.ParquetFile(path)
print("schema:", pf.schema_arrow)
print("rows:", pf.metadata.num_rows, "row groups:", pf.metadata.num_row_groups)
print("metadata:", {k.decode(): v.decode()[:300] for k, v in (pf.schema_arrow.metadata or {}).items()})
table = pf.read(columns=["id", "title", "raw", "vector"])
ids = table["id"].to_pylist()
vec = np.array(table["vector"].slice(0, n).to_pylist(), dtype=np.float64)
print("row0 id/title:", ids[0], table["title"][0], "|", table["raw"][0].as_py()[:120])
print("vector norms:", np.round(np.percentile(np.linalg.norm(vec, axis=1), [0, 50, 100]), 3))
sp = read_npy_rows(AKER + "/sphere-10m.npy", 0, n, 768, "<f8")
print("row-aligned cosine (sphere-10m[i] vs hermes[i]):",
      np.round(np.percentile(cosine_rows(sp, vec), [0, 5, 50, 100]), 4))
print("max abs diff row0:", float(np.abs(sp[0] - vec[0]).max()))
# nearest match by cosine over the head, to detect offsets or permutations
a = sp / np.linalg.norm(sp, axis=1, keepdims=True)
b = vec / np.linalg.norm(vec, axis=1, keepdims=True)
best = (a[:200] @ b.T).argmax(1)
print("best-match index == i for first 200:", float(np.mean(best == np.arange(200))))
print("best-match cos median:", float(np.median((a[:200] @ b.T).max(1))))
print(f"elapsed {time.time() - t0:.1f}s")
