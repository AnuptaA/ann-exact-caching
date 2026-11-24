"""Shared artifacts of a dataset: where they live, the manifest in the store, pull and push.

Artifacts are built once by the prepare stage and pulled by every run instance. A manifest per dataset
(artifacts/<dataset>/manifest.json in the store) lists, for each artifact, its local path, files with size and
sha256, the command that made it and the code commit.
"""

import json
from pathlib import Path

from recal import store as store_mod

NAMES = ("prep", "diskann_answers", "diskann_index", "shift_preps", "update_preps", "pgvector_db",
         "medrag_budgets")
# artifacts that are set up on the instance instead of pulled
SETUP_ONLY = ("pgvector_db",)


def workloads_of(cfg: dict, dataset: str) -> dict:
    return {n: w for n, w in cfg["workloads"].items() if w["dataset"] == dataset}


def datasets(cfg: dict) -> list[str]:
    return list(dict.fromkeys(w["dataset"] for w in cfg["workloads"].values()))


def local_paths(cfg: dict, dataset: str) -> dict[str, list[str]]:
    """artifact -> local paths (directories or files) in the repository layout of the configuration."""
    ws, q4 = workloads_of(cfg, dataset), cfg["q4"]
    out = {
        "prep": sorted({w["prep"] for w in ws.values()}),
        "diskann_answers": sorted({w["diskann_file"] for w in ws.values()}),
        "diskann_index": sorted({str(Path(w["diskann_file"]).parent / "diskann") for w in ws.values()}),
        "shift_preps": [f"{q4['shift_prep_dir']}/{n}_{kind}_seed{s}" for n in q4["shift_workloads"] if n in ws
                        for kind in ("abrupt", "gradual", "correlated") for s in cfg["shift_seeds"]],
        "update_preps": [f"{q4['update_prep_dir']}/{n}_{mode}{round(100 * rate):d}" for n in q4["update_workloads"]
                         if n in ws for mode in q4["update_modes"] for rate in q4["update_rates"]],
        "pgvector_db": [],
        "medrag_budgets": [cfg["medrag_budgets_file"]] if dataset == "medrag" and cfg.get("medrag_budgets_file") else [],
    }
    return out


def manifest_key(dataset: str) -> str:
    return store_mod.artifact_key(dataset, "manifest.json")


def read_manifest(store, dataset: str) -> dict:
    raw = store.backend.get_bytes(manifest_key(dataset))
    return json.loads(raw) if raw else {"dataset": dataset, "artifacts": {}}


def write_manifest(store, dataset: str, manifest: dict) -> None:
    store.backend.put_bytes(manifest_key(dataset), json.dumps(manifest, indent=1).encode())


def files_of(path: Path) -> list[Path]:
    path = Path(path)
    return [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())


def describe(name: str, paths: list[str], command: str, commit: str, root: Path = Path(".")) -> dict:
    """Manifest entry of an artifact: its paths, files (size, sha256), command and code commit."""
    entry_files = []
    for p in paths:
        for f in files_of(root / p):
            entry_files.append({"path": str(f.relative_to(root)), "size": f.stat().st_size,
                                "sha256": store_mod.sha256_file(f)})
    return {"name": name, "paths": paths, "files": entry_files, "command": command, "code_commit": commit,
            "bytes": sum(f["size"] for f in entry_files)}


def push_artifact(store, dataset: str, entry: dict, root: Path = Path(".")) -> int:
    """Upload the files of an artifact entry and record it in the dataset's manifest."""
    pushed = 0
    for f in entry["files"]:
        key = store_mod.artifact_key(dataset, f"{entry['name']}/{f['path']}")
        pushed += store.push_file(key, root / f["path"])
    manifest = read_manifest(store, dataset)
    manifest["artifacts"][entry["name"]] = entry
    write_manifest(store, dataset, manifest)
    return pushed


def pull_artifacts(store, wanted: list[str], root: Path = Path(".")) -> int:
    """wanted: '<dataset>/<artifact>' names. Files go to their manifest paths under root, verified on arrival."""
    pulled = 0
    by_dataset: dict[str, list[str]] = {}
    for item in wanted:
        dataset, name = item.split("/", 1)
        by_dataset.setdefault(dataset, []).append(name)
    for dataset, names in by_dataset.items():
        manifest = read_manifest(store, dataset)
        for name in names:
            if name in SETUP_ONLY:
                continue
            entry = manifest["artifacts"].get(name)
            if entry is None:
                raise FileNotFoundError(f"artifact {dataset}/{name} is not in the store manifest")
            for f in entry["files"]:
                key = store_mod.artifact_key(dataset, f"{name}/{f['path']}")
                pulled += store.pull_file(key, root / f["path"])
    return pulled
