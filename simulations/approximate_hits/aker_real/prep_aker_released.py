#!/usr/bin/env python3
"""
A prep folder for eval_aker_real.py built from Aker's RELEASED trace files only, to
reproduce the paper's setup exactly (context doc 9f): the request vectors as released
(TriviaQA: the bge-base query vectors, not re-embedded) and Aker's own ground truth
(gt_ids: top-100 by Euclidean distance over the 10M base, from the .pkl files) as the
backend's answers and as truth.

Writes <out>/meta.json (kind trace, the original workload name, so the base is
SPHERE / SPACEV), q.npy (distinct request vectors, float32), streams.npz ("trace": the
request sequence as indices into q), top500.npy (despite the name, 100 columns:
Aker's gt_ids; the harness reads only the first k + delta).

  .venv/bin/python simulations/approximate_hits/aker_real/prep_aker_released.py \
      --workload triviaqa-aker-0.99 --out simulations/approximate_hits/results/aker_released/triviaqa-aker-0.99
"""

import json
import pickle
import argparse
from pathlib import Path

import numpy as np

AKER = Path("datasets/aker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, help="e.g. triviaqa-aker-0.99, spacev-aker-0.6")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kind, _, z = args.workload.rpartition("-")
    stem = {"triviaqa-aker": "triviaqa", "spacev-aker": "spacev"}[kind]
    vecs = np.load(AKER / f"{stem}-sim-100k-{z}.npy")
    with open(AKER / f"{stem}-sim-100k-{z}.pkl", "rb") as f:
        trace = pickle.load(f)
    assert len(trace) == len(vecs)
    gt = np.stack([np.asarray(t["gt_ids"], dtype=np.int64) for t in trace])
    # check: the .pkl payloads are the .npy rows
    for i in (0, len(vecs) // 2, len(vecs) - 1):
        assert np.allclose(np.asarray(trace[i]["payload"], dtype=np.float32),
                           vecs[i].astype(np.float32), atol=1e-3), "payload mismatch"
    uniq, first, inv = np.unique(vecs, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    top = gt[first]
    # identical request vectors must have identical ground truth
    same = np.all(gt == top[inv], axis=1).mean()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"kind": "trace", "trace": args.workload,
                                               "source": "Aker released .npy + .pkl gt_ids"}))
    np.save(out / "q.npy", uniq.astype(np.float32))
    np.savez(out / "streams.npz", trace=inv.astype(np.int64))
    np.save(out / "top500.npy", top.astype(np.int32))
    info = {"requests": int(len(vecs)), "distinct": int(len(uniq)), "gt_width": int(gt.shape[1]),
            "share_requests_with_identical_gt_for_identical_vectors": float(same)}
    (out / "prep_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
