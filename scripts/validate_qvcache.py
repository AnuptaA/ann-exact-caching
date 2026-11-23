#!/usr/bin/env python3
"""Validate the QVCache setup against the authors' artifact (runs on the cluster, Linux).

  python -m scripts.validate_qvcache --artifact-dir $TP/qvcache --out validation.json \\
      [--command python/scripts/benchmark/qvcache_benchmark_faiss_backend.sh]

Stages, each written to the report:
  artifact          the artifact's own benchmark script as provided (siftsmall sample); its window_metrics lines
                    give hit_ratio, recall_all and recall_cache_hits
  adapter           the same data and the script's own parameters through our adapter and driver
                    (recal/qvcache_driver.py): hit rate and recall must match the artifact's within --tolerance
  reimplementation  the earlier simulator (online_baselines.QVCache) on the same stream, differences recorded
                    only (it produces no reported number)
The artifact uses asynchronous inserts, so hit rates vary a little between runs: the tolerance is a band, not equality.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

VARIABLE = re.compile(r"^([A-Z][A-Z0-9_]*)=(\"[^\"]*\"|[^\s#]+)", re.M)
SCRIPT_TO_ARG = {"R": "R", "MEMORY_L": "memory_L", "B": "B", "M": "M", "ALPHA": "alpha", "P": "p",
                 "DEVIATION_FACTOR": "deviation_factor", "MEMORY_INDEX_MAX_POINTS": "memory_index_max_points",
                 "PCA_DIM": "pca_dim", "BUCKETS_PER_DIM": "buckets_per_dim",
                 "NUMBER_OF_MINI_INDEXES": "number_of_mini_indexes"}


def script_variables(text: str) -> dict:
    """NAME=value assignments of a benchmark shell script, numbers converted."""
    out = {}
    for name, value in VARIABLE.findall(text):
        value = value.strip('"')
        try:
            out[name] = int(value)
        except ValueError:
            try:
                out[name] = float(value)
            except ValueError:
                out[name] = value
    return out


def parse_window_metrics(text: str) -> dict:
    """Mean hit ratio and recalls over the window_metrics JSON lines of the artifact's output."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"window_metrics"' in line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise ValueError("no window_metrics lines in the artifact output")

    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {"windows": len(rows), "hit_ratio": mean("hit_ratio"), "recall_all": mean("recall_all"),
            "recall_cache_hits": mean("recall_cache_hits")}


def compare(artifact: dict, ours: dict, tolerance: float) -> dict:
    """Differences of hit rate and recall; ok when each is within the tolerance."""
    diff = {"hit_ratio": ours["hit_ratio"] - artifact["hit_ratio"],
            "recall_all": ours["recall_all"] - artifact["recall_all"]}
    return {"differences": diff, "tolerance": tolerance, "ok": all(abs(v) <= tolerance for v in diff.values())}


def run_artifact(artifact_dir: Path, command: str) -> dict:
    out = subprocess.run(["bash", command], cwd=artifact_dir, capture_output=True, text=True, check=True).stdout
    return {"command": command, "metrics": parse_window_metrics(out),
            "variables": script_variables((artifact_dir / command).read_text())}


def run_adapter(artifact_dir: Path, variables: dict, qvc=None) -> dict:
    """The script's data and parameters through recal.qvcache_driver (the fixed stream, K from the script)."""
    from recal import qvcache_driver as qd
    from recal.backend import ExactBackend
    from recal.qvcache_backends import StoredBackend
    from recal.records import Records
    qvc = qvc or qd.load_module()
    data = artifact_dir / "data" / variables["DATASET"]
    base, _, _ = qvc.load_aligned_binary_data(str(data / f"{variables['DATASET']}_base.bin"))
    queries, _, _ = qvc.load_aligned_binary_data(str(variables["QUERY_PATH"]) if "QUERY_PATH" in variables else
                                                 str(data / f"{variables['DATASET']}_query.bin"))
    truth, _ = qvc.load_ground_truth_data(str(variables.get("GROUNDTRUTH_PATH",
                                                            data / f"{variables['DATASET']}_groundtruth.bin")))
    conf = qd.settings()
    for name, key in SCRIPT_TO_ARG.items():
        if name in variables and key in conf:
            conf[key] = variables[name]
    K = int(variables.get("K", 10))
    backend = StoredBackend(ExactBackend(base))
    bin_path = data / f"{variables['DATASET']}_base.bin"
    cache = qd.make_cache(qvc, bin_path, artifact_dir / "validation_pca", backend, conf,
                          float(variables.get("DEVIATION_FACTOR", 0.25)),
                          int(variables.get("MEMORY_INDEX_MAX_POINTS", 100000)))
    ks = np.full(len(queries), K, dtype=np.int32)
    records = Records()
    totals = qd.replay(cache, backend, queries, ks, 0, lambda pos, k: truth[pos, :k], records)
    return {"hit_ratio": totals["hits"] / len(queries), "recall_all": totals["recall_sum"] / len(queries),
            "queries": len(queries)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--command", default="python/scripts/benchmark/qvcache_benchmark_faiss_backend.sh")
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    report = {"artifact": run_artifact(a.artifact_dir, a.command)}
    report["adapter"] = run_adapter(a.artifact_dir, report["artifact"]["variables"])
    report["comparison"] = compare(report["artifact"]["metrics"], report["adapter"], a.tolerance)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["comparison"]))
    return 0 if report["comparison"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
