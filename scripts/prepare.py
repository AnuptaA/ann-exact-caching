#!/usr/bin/env python3
"""Prepare a dataset: build its shared artifacts once and push them, with a manifest, to the store.

  python -m scripts.prepare --config experiments/matrix.json --dataset spacev [--dry-run] [--store s3://b/p]
  python -m scripts.prepare --config experiments/matrix.json --dataset spacev --check   # a finished prepare is intact

Stages (recal/prepare.py): fetch and verify, request streams and exact answers, DiskANN index and answer tables,
pgvector database (update workloads), shift streams and update schedules, MedRAG working set, QVCache memory
measurement. Each artifact is listed in artifacts/<dataset>/manifest.json with path, size, sha256, command and code
commit. TripClick-derived artifacts are not uploaded unless --upload-restricted is given.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recal import artifacts, prepare, store as store_mod  # noqa: E402
from recal.measure import jobs_util  # noqa: E402


GIB = 2 ** 30
MIN_FREE_GB = 5


def run_commands(commands: list[str]) -> None:
    for cmd in commands:
        if subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=False).returncode != 0:
            raise SystemExit(f"stage command failed: {cmd}")


def marker_path(root: Path, dataset: str, stage) -> Path:
    return root / "results" / ".prepare" / dataset / f"{stage.name}.json"


def commands_hash(stage) -> str:
    return hashlib.sha256("\n".join(stage.commands).encode()).hexdigest()


def stage_done(root: Path, dataset: str, stage, cfg: dict) -> bool:
    """A stage with output paths is done when its marker has the same commands and every path exists."""
    paths = stage.paths or []
    marker = marker_path(root, dataset, stage)
    if not paths or not marker.exists():
        return False
    try:
        recorded = json.loads(marker.read_text()).get("commands")
    except ValueError:
        return False
    return recorded == commands_hash(stage) and all((root / p).exists() for p in paths)


def mark_done(root: Path, dataset: str, stage) -> None:
    marker = marker_path(root, dataset, stage)
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps({"commands": commands_hash(stage)}))
    tmp.replace(marker)


def check_free_disk(root: Path, stage, cfg: dict, dataset: str) -> None:
    """Refuse a stage that cannot fit; the DiskANN build needs about twice the base vectors."""
    need = MIN_FREE_GB * GIB
    if stage.name == "diskann_index":
        base = root / prepare.settings(cfg, dataset)["base"]
        if base.exists():
            need = max(need, 2 * base.stat().st_size + 2 * GIB)
    free = shutil.disk_usage(root).free
    if free < need:
        raise SystemExit(f"stage {stage.name} needs about {need / GIB:.0f} GiB free in {root.resolve()}; "
                         f"{free / GIB:.0f} GiB are free")


def push_stage(s, cfg: dict, dataset: str, stage, root: Path, upload: bool) -> dict | None:
    if stage.artifact is None:
        return None
    paths = stage.paths or artifacts.local_paths(cfg, dataset)[stage.artifact]
    entry = artifacts.describe(stage.artifact, paths, " && ".join(stage.commands), jobs_util.code_commit(), root)
    if s is not None and upload:
        artifacts.push_artifact(s, dataset, entry, root)
    return entry


def prepare_dataset(cfg: dict, dataset: str, s=None, dry_run: bool = False, upload_restricted: bool = False,
                    root: Path = Path("."), force: bool = False) -> dict:
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
        if not force and stage_done(root, dataset, stage, cfg):
            print(f"[{dataset}] {stage.name}: done, skipped (--force redoes it)")
        else:
            check_free_disk(root, stage, cfg, dataset)
            run_commands(stage.commands)
            if stage.paths:
                mark_done(root, dataset, stage)
        entry = push_stage(s, cfg, dataset, stage, root, upload)
        if entry is not None:
            manifest["artifacts"][stage.artifact] = entry
    return manifest


def check_dataset(cfg: dict, dataset: str, manifest_dir: Path, root: Path = Path(".")) -> list[tuple[bool, str]]:
    """One (ok, line) per expected artifact of a finished prepare: it is in the manifest and every file exists with
    its recorded size and checksum. Nothing is changed."""
    manifest_file = manifest_dir / f"{dataset}.json"
    if not manifest_file.exists():
        return [(False, f"{dataset}: no manifest {manifest_file} -> run make prepare DATASET={dataset}")]
    manifest = json.loads(manifest_file.read_text())
    expected = [st.artifact for st in prepare.stages(cfg, dataset) if st.artifact] if dataset != "tiny" else \
        ["prep", "diskann_answers"]
    out = []
    for name in expected:
        entry = manifest.get("artifacts", {}).get(name)
        if entry is None:
            out.append((False, f"{dataset}/{name}: not in the manifest -> rerun make prepare DATASET={dataset}"))
            continue
        problems = []
        for f in entry["files"]:
            path = root / f["path"]
            if not path.exists():
                problems.append(f"{f['path']} is missing")
            elif path.stat().st_size != f["size"] or store_mod.sha256_file(path) != f["sha256"]:
                problems.append(f"{f['path']} differs from the manifest")
        if problems:
            out.append((False, f"{dataset}/{name}: {problems[0]} ({len(problems)} problem(s)) -> rerun make prepare "
                               f"DATASET={dataset} (add --force to redo it)"))
        else:
            out.append((True, f"{dataset}/{name}: {len(entry['files'])} files, {entry['bytes'] / 1e9:.2f} GB"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("experiments/matrix.json"))
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--store", default=None)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--upload-restricted", action="store_true", help="also upload TripClick-derived artifacts")
    ap.add_argument("--dry-run", action="store_true", help="print the stage commands only")
    ap.add_argument("--force", action="store_true", help="redo stages that are marked done")
    ap.add_argument("--check", action="store_true",
                    help="verify a finished prepare (every artifact of the manifest exists with its checksum)")
    ap.add_argument("--out", type=Path, default=Path("results/manifests"))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    names = artifacts.datasets(cfg) if a.dataset == "all" else [a.dataset]
    if a.check:
        failed = False
        for dataset in names:
            for ok, line in check_dataset(cfg, dataset, a.out):
                print(("PASS  " if ok else "FAIL  ") + line)
                failed = failed or not ok
        return 1 if failed else 0
    s = None if a.no_upload or a.dry_run else store_mod.open_store(a.store)
    for dataset in names:
        manifest = prepare_dataset(cfg, dataset, s, a.dry_run, a.upload_restricted, force=a.force)
        if not a.dry_run:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f"{dataset}.json").write_text(json.dumps(manifest, indent=1))
            total = sum(e["bytes"] for e in manifest["artifacts"].values())
            print(f"{dataset}: {len(manifest['artifacts'])} artifacts, {total / 1e9:.2f} GB, "
                  f"uploaded: {manifest['uploaded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
