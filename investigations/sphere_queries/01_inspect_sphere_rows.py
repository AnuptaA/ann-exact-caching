"""Part A step 3: representability, norms, mean vector and outlier dims of sphere-10m rows."""

import sys
import time

import numpy as np

from common import AKER, range_get

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0

t0 = time.time()
buf = range_get(AKER + "/sphere-10m.npy", 128 + START * 768 * 8,
                128 + (START + N_ROWS) * 768 * 8 - 1)
x = np.frombuffer(buf, dtype="<f8").reshape(N_ROWS, 768)
print(f"read {N_ROWS} rows from {START} in {time.time() - t0:.1f}s")
print("float32 exact:", float(np.mean(x.astype(np.float32).astype(np.float64) == x)))
print("float16 exact:", float(np.mean(x.astype(np.float16).astype(np.float64) == x)))
norms = np.linalg.norm(x, axis=1)
print("norm min/p5/median/p95/max:", np.round(np.percentile(norms, [0, 5, 50, 95, 100]), 3))
mean = x.mean(0)
print("mean vector norm:", round(float(np.linalg.norm(mean)), 3),
      "mean|x| per-dim:", round(float(np.abs(x).mean()), 4))
sd = x.std(0)
top = np.argsort(-sd)[:5]
print("top std dims:", top.tolist(), np.round(sd[top], 3).tolist(), "median std", round(float(np.median(sd)), 4))
topm = np.argsort(-np.abs(mean))[:5]
print("top |mean| dims:", topm.tolist(), np.round(mean[topm], 3).tolist())
print("cos to mean median:", round(float(np.median(x @ mean / (norms * np.linalg.norm(mean)))), 3))
print("first row[:6]:", x[0, :6].tolist())
