#!/usr/bin/env python3
"""Fetch the TripClick and MedRAG files from the shared Google Drive folder.

Only the allowed files are downloaded, by Drive file id, never the whole folder. Run it through
scripts/fetch_drive_data.sh. Modes:
  --local    TripClick files and MedRAG's small files (no corpus, no PMID arrays)
  --cluster  TripClick and all of MedRAG (about 74 GB of corpus; Linux cluster only)
  --dry-run  list the actions without downloading anything
"""

import json
import shutil
import argparse
from pathlib import Path

FOLDER_ID = "<RECAL_DRIVE_ESCI>"
MIN_FREE_GB = 10

# (drive directory, file name) -> destination under the repository root
TRIPCLICK = {
    "tripclick-ip-docs.npy": "datasets/aker/tripclick-ip-docs.npy",
    "tripclick-ip-requests-2018-01-01.npy": "datasets/aker/tripclick-ip-requests-2018-01-01.npy",
    "requests-2018-01-01.json": "datasets/tripclick/requests-2018-01-01.json",
}
MEDRAG_LOCAL = {
    "medrag-ip-requests.npy": "datasets/aker/medrag-ip-requests.npy",
    "base.json": "datasets/medrag/base.json",
    "requests.json": "datasets/medrag/requests.json",
    "zipf_draw.json": "datasets/medrag/zipf_draw.json",
    "paraphrases.json": "datasets/medrag/paraphrases.json",
    "paraphrase_stats.json": "datasets/medrag/paraphrase_stats.json",
    "benchmark.json": "datasets/medrag/benchmark.json",
}
MEDRAG_CLUSTER = {
    "medrag-ip-docs.npy": "datasets/aker/medrag-ip-docs.npy",
    "base_pmids.npy": "datasets/medrag/base_pmids.npy",
    "medrag_pmids.npy": "datasets/medrag/medrag_pmids.npy",
}


def allowed(mode: str) -> dict[tuple[str, str], str]:
    """Map (drive directory, file name) to the destination for the chosen mode."""
    wanted = {("tripclick", name): dest for name, dest in TRIPCLICK.items()}
    medrag = dict(MEDRAG_LOCAL, **(MEDRAG_CLUSTER if mode == "cluster" else {}))
    wanted.update({("medrag", name): dest for name, dest in medrag.items()})
    return wanted


def select(listing: list[dict], mode: str) -> list[dict]:
    """Keep the listed files that are allowed. Each item has path (relative, with '/'), id and size."""
    wanted = allowed(mode)
    chosen = []
    for item in listing:
        parts = item["path"].split("/")
        key = (parts[-2], parts[-1]) if len(parts) >= 2 else ("", parts[-1])
        if key in wanted:
            chosen.append(dict(item, destination=wanted[key]))
    return chosen


def plan(chosen: list[dict], root: Path) -> list[dict]:
    """Skip files already on disk with the same size. A different size is replaced by the Drive version."""
    actions = []
    for item in chosen:
        dest = root / item["destination"]
        size = item.get("size")
        if dest.exists() and size is not None and dest.resolve().stat().st_size == size:
            actions.append(dict(item, action="keep"))
        elif dest.exists():
            actions.append(dict(item, action="replace" if size is not None else "download"))
        else:
            actions.append(dict(item, action="download"))
    return actions


def enough_space(actions: list[dict], free_bytes: int, floor_gb: float = MIN_FREE_GB) -> bool:
    need = sum(a.get("size") or 0 for a in actions if a["action"] != "keep")
    return free_bytes - need >= floor_gb * 2**30


def list_drive(folder_id: str) -> list[dict]:
    import gdown
    files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True, use_cookies=False)
    return [{"path": f.path, "id": f.id, "size": getattr(f, "size", None)} for f in files]


def download(item: dict, staging: Path) -> Path:
    import gdown
    target = staging / Path(item["destination"]).name
    gdown.download(id=item["id"], output=str(target), quiet=False)
    return target


def place(staged: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    shutil.move(str(staged), str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--cluster", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--staging", type=Path, default=Path("datasets/.drive_staging"))
    parser.add_argument("--folder-id", default=FOLDER_ID)
    parser.add_argument("--listing", type=Path, help="JSON list of {path, id, size} instead of asking Drive (offline dry run)")
    args = parser.parse_args()
    listing = json.loads(args.listing.read_text()) if args.listing else list_drive(args.folder_id)
    chosen = select(listing, "cluster" if args.cluster else "local")
    actions = plan(chosen, args.root)
    free = shutil.disk_usage(args.root).free
    for a in actions:
        print(f"{a['action']:9s} {a['path']} -> {a['destination']} ({a.get('size')} bytes)")
    if args.dry_run:
        return 0
    if not enough_space(actions, free):
        print(f"refusing: fewer than {MIN_FREE_GB} GB would remain free")
        return 2
    args.staging.mkdir(parents=True, exist_ok=True)
    for a in actions:
        if a["action"] != "keep":
            place(download(a, args.staging), args.root / a["destination"])
    shutil.rmtree(args.staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
