#!/usr/bin/env python3
"""Prepared inputs of the micro-experiments in the store, so a fresh instance pulls them instead of preparing.

  python -m scripts.micro.inputs push   --dataset spacev     # from the instance that prepared it
  python -m scripts.micro.inputs pull   --dataset spacev     # on a fresh instance
  python -m scripts.micro.inputs verify --dataset spacev     # hashes of the local files against the manifest

Per workload (the ones scripts.micro.run runs): meta.json, q.npy, top500.npy, streams.npz of its prep folder, and
for MedRAG the working-set budgets file, under inputs/micro/<dataset>/<workload>/ with a manifest of sha256 hashes.
The corpus itself is not copied (inputs/<dataset>/ holds it for MedRAG and LAION; SPACEV and TriviaQA fetch it).
The store is derived from RECAL_STORE: s3://bucket/... becomes s3://bucket/inputs/micro, a directory D becomes
D/inputs/micro.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recal import store as store_mod  # noqa: E402
from recal.micro import provenance  # noqa: E402
from scripts.micro.run import workloads_for  # noqa: E402

FILES = ("meta.json", "q.npy", "top500.npy", "streams.npz")
MANIFEST = "manifest.json"


def inputs_store(url: str | None):
    import os
    url = url or os.environ.get("RECAL_STORE")
    if url is None:
        raise SystemExit("set RECAL_STORE (s3://bucket/prefix or a directory) or pass --store")
    if url.startswith("s3://"):
        bucket = url[5:].partition("/")[0]
        return store_mod.open_store(f"s3://{bucket}/inputs/micro")
    return store_mod.open_store(str(Path(url) / "inputs" / "micro"))


def files_of(cfg: dict, name: str, w: dict) -> dict[str, Path]:
    out = {f: Path(w["prep"]) / f for f in FILES}
    if w["dataset"] == "medrag" and cfg.get("medrag_budgets_file"):
        out["medrag_budgets.json"] = Path(cfg["medrag_budgets_file"])
    return out


def push(store, dataset: str, cfg: dict) -> int:
    n = 0
    for name, w in workloads_for(cfg, dataset).items():
        files = files_of(cfg, name, w)
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise SystemExit(f"cannot push {name}: missing {', '.join(missing)}; run make prepare first")
        manifest = {f: provenance.file_digest(p) for f, p in files.items()}
        for f, p in files.items():
            n += int(store.push_file(f"{dataset}/{name}/{f}", p))
        tmp = Path(w["prep"]) / f".{MANIFEST}"
        tmp.write_text(json.dumps(manifest, indent=1))
        n += int(store.push_file(f"{dataset}/{name}/{MANIFEST}", tmp))
        tmp.unlink()
        print(f"[{name}] pushed {len(files)} inputs ({sum(p.stat().st_size for p in files.values()) / 1e6:.0f} MB)")
    return n


def manifest_of(store, dataset: str, name: str) -> dict:
    tmp = Path(".micro_manifest.tmp")
    try:
        store.pull_file(f"{dataset}/{name}/{MANIFEST}", tmp)
        return json.loads(tmp.read_text())
    finally:
        tmp.unlink(missing_ok=True)


def verify(store, dataset: str, cfg: dict) -> None:
    for name, w in workloads_for(cfg, dataset).items():
        manifest = manifest_of(store, dataset, name)
        for f, p in files_of(cfg, name, w).items():
            if not p.exists():
                raise SystemExit(f"{p}: missing; run python -m scripts.micro.inputs pull --dataset {dataset}")
            if provenance.file_digest(p) != manifest.get(f):
                raise SystemExit(f"{p}: sha256 differs from the manifest in the store; pull again or rerun prepare")


def pull(store, dataset: str, cfg: dict) -> int:
    n = 0
    for name, w in workloads_for(cfg, dataset).items():
        for f, p in files_of(cfg, name, w).items():
            n += int(store.pull_file(f"{dataset}/{name}/{f}", p))
        print(f"[{name}] inputs are in place under {w['prep']}")
    verify(store, dataset, cfg)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("push", "pull", "verify"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", type=Path, default=Path("experiments/matrix.json"))
    ap.add_argument("--store", default=None)
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    store = inputs_store(a.store)
    if a.action == "push":
        push(store, a.dataset, cfg)
    elif a.action == "pull":
        pull(store, a.dataset, cfg)
    else:
        verify(store, a.dataset, cfg)
        print(f"{a.dataset}: inputs match the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
