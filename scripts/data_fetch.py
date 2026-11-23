#!/usr/bin/env python3
"""Fetch the evaluation datasets, one mode per dataset.

Run it through scripts/fetch_data.sh. Datasets:
  --spacev    SPACEV base vectors and Aker's request streams (Zipf 0.3, 0.6, 0.99)
  --triviaqa  TriviaQA requests re-embedded with SPHERE's DPR query encoder
              (inner-product form); the base vectors come with --sphere
  --sphere    SPHERE inner-product base vectors (10M x 769)
  --medrag    MedRAG vectors and workload (--no-corpus skips the 74 GB corpus)
  --tripclick TripClick vectors and requests
  --esci      ESCI files for local checks (never part of --all)
  --all       every dataset above except ESCI
  --list      print what each mode would fetch (no network)
  --dry-run   print the actions and the space check without downloading

SPACEV streams come from Aker's Hugging Face release (dataset sjoon-oh/aker).
The other files come from the shared Google Drive folder, by file name, never the
whole folder. Free space is checked before every download.
"""

import json
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts import drive_fetch

HF_BASE = "https://huggingface.co/datasets/sjoon-oh/aker/resolve/main"
ZIPF = ("0.3", "0.6", "0.99")
MIN_FREE_GB = 10
NPY_HEADER = 128


@dataclass(frozen=True)
class Item:
    source: str
    name: str
    dest: str
    size: int | None = None
    drive_dir: str | None = None


def npy_size(rows: int, dim: int, itemsize: int) -> int:
    return rows * dim * itemsize + NPY_HEADER


def spacev_items() -> list[Item]:
    items = [Item("hf", "spacev-10m.npy", "datasets/aker/spacev-10m.npy",
                  npy_size(10_000_000, 100, 1))]
    for zipf in ZIPF:
        name = f"spacev-sim-100k-{zipf}.npy"
        items.append(Item("hf", name, f"datasets/aker/{name}",
                          npy_size(100_000, 100, 1)))
    return items


def triviaqa_items() -> list[Item]:
    items = []
    for zipf in ZIPF:
        name = f"triviaqa-ip-sim-100k-{zipf}.npy"
        items.append(Item("drive", name, f"datasets/aker/{name}",
                          npy_size(100_000, 769, 4)))
    return items


def sphere_items() -> list[Item]:
    return [
        Item("drive", "sphere-ip-10m.npy", "datasets/aker/sphere-ip-10m.npy",
             npy_size(10_000_000, 769, 4)),
        Item("drive", "sphere-ip-10m.json", "datasets/aker/sphere-ip-10m.json"),
    ]


def medrag_items(corpus: bool = True) -> list[Item]:
    files = dict(drive_fetch.MEDRAG_LOCAL)
    if corpus:
        files.update(drive_fetch.MEDRAG_CLUSTER)
    return [Item("drive", name, dest, drive_dir="medrag") for name, dest in files.items()]


def tripclick_items() -> list[Item]:
    return [Item("drive", name, dest, drive_dir="tripclick")
            for name, dest in drive_fetch.TRIPCLICK.items()]


def esci_items(listing: list[dict]) -> list[dict]:
    """Every file under the Drive ESCI directory, placed under datasets/esci."""
    chosen = []
    for entry in listing:
        parts = entry["path"].split("/")
        if "ESCI" in parts[:-1]:
            rel = "/".join(parts[parts.index("ESCI") + 1:])
            chosen.append(dict(entry, destination=f"datasets/esci/{rel}"))
    return chosen


MODES = ("spacev", "triviaqa", "sphere", "medrag", "tripclick")


def items_for(mode: str, corpus: bool = True) -> list[Item]:
    if mode == "medrag":
        return medrag_items(corpus)
    return {"spacev": spacev_items, "triviaqa": triviaqa_items,
            "sphere": sphere_items, "tripclick": tripclick_items}[mode]()


def match_drive(item: Item, listing: list[dict]) -> dict | None:
    for entry in listing:
        parts = entry["path"].split("/")
        if parts[-1] != item.name:
            continue
        if item.drive_dir is None or (len(parts) >= 2 and parts[-2] == item.drive_dir):
            return entry
    return None


def action_for(dest: Path, expected: int | None, force: bool) -> str:
    if not dest.exists():
        return "download"
    if force:
        return "replace"
    if expected is not None and dest.resolve().stat().st_size != expected:
        return "replace"
    return "keep"


def build_actions(modes: list[str], root: Path, listing: list[dict] | None,
                  corpus: bool, force: bool, esci: bool) -> list[dict]:
    actions = []
    for mode in modes:
        for item in items_for(mode, corpus):
            entry = None
            size = item.size
            if item.source == "drive":
                entry = match_drive(item, listing or [])
                if entry is None and listing is not None:
                    actions.append({"mode": mode, "item": item, "action": "missing",
                                    "size": size})
                    continue
                if entry is not None and entry.get("size"):
                    size = entry["size"]
            actions.append({"mode": mode, "item": item, "entry": entry, "size": size,
                            "action": action_for(root / item.dest, size, force)})
    if esci:
        for entry in esci_items(listing or []):
            item = Item("drive", Path(entry["destination"]).name, entry["destination"])
            actions.append({"mode": "esci", "item": item, "entry": entry,
                            "size": entry.get("size"),
                            "action": action_for(root / item.dest, entry.get("size"), force)})
    return actions


def total_needed(actions: list[dict]) -> int:
    return sum(a["size"] or 0 for a in actions if a["action"] in ("download", "replace"))


def space_ok(need: int, free: int, floor_gb: float = MIN_FREE_GB) -> bool:
    return free - need >= floor_gb * 2**30


def download_item(action: dict, root: Path, staging: Path) -> None:
    item = action["item"]
    dest = root / item.dest
    if item.source == "hf":
        from workloads.q1.datasets import common
        dest.parent.mkdir(parents=True, exist_ok=True)
        common.download_file(f"{HF_BASE}/{item.name}", dest)
        return
    drive_fetch.place(drive_fetch.download(
        dict(action["entry"], destination=item.dest), staging), dest)


def print_actions(actions: list[dict]) -> None:
    for a in actions:
        item = a["item"]
        size = a["size"] if a["size"] is not None else "?"
        print(f"{a['action']:9s} {a['mode']:9s} {item.source:5s} {item.name} "
              f"-> {item.dest} ({size} bytes)")


def needs_drive(modes: list[str], esci: bool) -> bool:
    return esci or any(i.source == "drive" for m in modes for i in items_for(m))


def print_listing() -> None:
    for mode in MODES:
        for item in items_for(mode):
            print(f"{mode:9s} {item.source:5s} {item.name} -> {item.dest}")
    print("esci      drive  every file under the Drive ESCI directory -> datasets/esci/ "
          "(--esci only)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for mode in MODES + ("esci", "all", "list"):
        parser.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--no-corpus", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--staging", type=Path, default=Path("datasets/.drive_staging"))
    parser.add_argument("--folder-id", default=drive_fetch.FOLDER_ID)
    parser.add_argument("--listing", type=Path,
                        help="JSON list of {path, id, size} instead of asking Drive")
    return parser.parse_args(argv)


def selected_modes(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(MODES)
    return [m for m in MODES if getattr(args, m)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print_listing()
        return 0
    modes = selected_modes(args)
    if not modes and not args.esci:
        print("choose a dataset (--spacev --triviaqa --sphere --medrag --tripclick "
              "--esci) or --all")
        return 2
    listing = None
    if needs_drive(modes, args.esci):
        if args.listing:
            listing = json.loads(args.listing.read_text())
        else:
            listing = drive_fetch.list_drive(args.folder_id)
    actions = build_actions(modes, args.root, listing, not args.no_corpus,
                            args.force, args.esci)
    print_actions(actions)
    if any(a["action"] == "missing" for a in actions):
        print("some files are not in the Drive listing")
        if not args.dry_run:
            return 3
    if args.dry_run:
        free = shutil.disk_usage(args.root).free
        ok = space_ok(total_needed(actions), free)
        print(f"needs {total_needed(actions)} bytes, free {free}, "
              f"{'enough' if ok else 'NOT enough'} with a {MIN_FREE_GB} GB floor")
        return 0
    args.staging.mkdir(parents=True, exist_ok=True)
    for a in actions:
        if a["action"] not in ("download", "replace"):
            continue
        free = shutil.disk_usage(args.root).free
        if not space_ok(a["size"] or 0, free):
            print(f"refusing {a['item'].name}: fewer than {MIN_FREE_GB} GB would "
                  "remain free")
            return 2
        download_item(a, args.root, args.staging)
    shutil.rmtree(args.staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
