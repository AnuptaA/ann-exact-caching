#!/usr/bin/env python3
"""Prepare a dataset: build its shared artifacts once and push them, with a manifest, to the store.

  python -m scripts.prepare --config experiments/matrix.json --dataset spacev [--dry-run] [--store s3://b/p]

Stages (recal/prepare.py): fetch and verify, request streams and exact answers, DiskANN index and answer tables,
pgvector database (update workloads), shift streams and update schedules, MedRAG working set, QVCache memory
measurement. Each artifact is listed in artifacts/<dataset>/manifest.json with path, size, sha256, command and code
commit. TripClick-derived artifacts are not uploaded unless --upload-restricted is given.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recal import artifacts, prepare, store as store_mod  # noqa: E402
from recal.measure import jobs_util  # noqa: E402


def run_commands(commands: list[str]) -> None:
    for cmd in commands:
        if subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=False).returncode != 0:
            raise SystemExit(f"stage command failed: {cmd}")


def push_stage(s, cfg: dict, dataset: str, stage, root: Path, upload: bool) -> dict | None:
    if stage.artifact is None:
        return None
    paths = stage.paths or artifacts.local_paths(cfg, dataset)[stage.artifact]
    entry = artifacts.describe(stage.artifact, paths, " && ".join(stage.commands), jobs_util.code_commit(), root)
    if s is not None and upload:
        artifacts.push_artifact(s, dataset, entry, root)
    return entry


def prepare_dataset(cfg: dict, dataset: str, s=None, dry_run: bool = False, upload_restricted: bool = False,
                    root: Path = Path(".")) -> dict:
    """Run every stage of the dataset; returns its manifest (also pushed when a store is given)."""
    upload = dataset not in prepare.RESTRICTED or upload_restricted
    manifest = {"dataset": dataset, "artifacts": {}, "uploaded": bool(s is not None and upload)}
    if dataset == "tiny":
        if dry_run:
            return manifest
        prepare.make_tiny(cfg, root)
        for name in ("prep", "diskann_answers"):
            stage = prepare.Stage(name, name, ["make_tiny"], artifacts.local_paths(cfg, "tiny")[name])
            manifest["artifacts"][name] = push_stage(s, cfg, dataset, stage, root, upload)
        return manifest
    for stage in prepare.stages(cfg, dataset):
        if dry_run:
            print(f"[{dataset}] {stage.name}" + (f" -> {stage.artifact}" if stage.artifact else ""))
            for cmd in stage.commands:
                print(f"  {cmd}")
            continue
        run_commands(stage.commands)
        entry = push_stage(s, cfg, dataset, stage, root, upload)
        if entry is not None:
            manifest["artifacts"][stage.artifact] = entry
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--store", default=None)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--upload-restricted", action="store_true", help="also upload TripClick-derived artifacts")
    ap.add_argument("--dry-run", action="store_true", help="print the stage commands only")
    ap.add_argument("--out", type=Path, default=Path("results/manifests"))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    names = artifacts.datasets(cfg) if a.dataset == "all" else [a.dataset]
    s = None if a.no_upload or a.dry_run else store_mod.open_store(a.store)
    for dataset in names:
        manifest = prepare_dataset(cfg, dataset, s, a.dry_run, a.upload_restricted)
        if not a.dry_run:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f"{dataset}.json").write_text(json.dumps(manifest, indent=1))
            total = sum(e["bytes"] for e in manifest["artifacts"].values())
            print(f"{dataset}: {len(manifest['artifacts'])} artifacts, {total / 1e9:.2f} GB, "
                  f"uploaded: {manifest['uploaded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
