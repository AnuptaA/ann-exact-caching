#!/usr/bin/env python3
"""Fetch the evaluation datasets, one mode per dataset.

Run it through scripts/fetch_data.sh. Datasets:
  --spacev    SPACEV base vectors and Aker's request streams (Zipf 0.3, 0.6, 0.99), Hugging Face sjoon-oh/aker
  --triviaqa  TriviaQA requests re-embedded with SPHERE's DPR query encoder (inner-product form), Drive
  --sphere    SPHERE base (Hugging Face sphere-10m.npy), then the inner-product corpus built from it; the M
              of the Drive file sphere-ip-10m.json must match
  --medrag    MedRAG corpus (MedRAG folder) and the 100K request stream (its own folder); --no-corpus skips
              the 74 GB corpus; --medrag-10k adds the 10K request files (a different rewording model; never
              used by the experiments)
  --tripclick TripClick vectors and requests
  --esci      ESCI files for local checks (never part of --all)
  --all       every dataset above except ESCI
  --list      print what each mode would fetch (no network)
  --dry-run   print the actions and the space check without downloading

Every Drive dataset has its own folder (FOLDERS). A file expected in a folder but missing from its listing stops
the run before anything is downloaded. Free space is checked before every download.
"""

import json
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/sjoon-oh/aker/resolve/main"
ZIPF = ("0.3", "0.6", "0.99")
MIN_FREE_GB = 10
NPY_HEADER = 128

FOLDERS = {
    "medrag": "<RECAL_DRIVE_MEDRAG>",
    "medrag100k": "<RECAL_DRIVE_MEDRAG_100K>",
    "tripclick": "<RECAL_DRIVE_TRIPCLICK>",
    "triviaqa": "<RECAL_DRIVE_TRIVIAQA>",
    "esci": "<RECAL_DRIVE_ESCI>",
}
TRIVIAQA_COUNTS = {"0.3": 99_903, "0.6": 99_990, "0.99": 99_907}


@dataclass(frozen=True)
class Item:
    source: str
    name: str
    dest: str
    size: int | None = None
    folder: str | None = None


def npy_size(rows: int, dim: int, itemsize: int) -> int:
    return rows * dim * itemsize + NPY_HEADER


def spacev_items() -> list[Item]:
    items = [Item("hf", "spacev-10m.npy", "datasets/aker/spacev-10m.npy", npy_size(10_000_000, 100, 1))]
    for zipf in ZIPF:
        name = f"spacev-sim-100k-{zipf}.npy"
        items.append(Item("hf", name, f"datasets/aker/{name}", npy_size(100_000, 100, 1)))
    return items


def triviaqa_items() -> list[Item]:
    items = [Item("drive", f"triviaqa-ip-sim-100k-{z}.npy", f"datasets/aker/triviaqa-ip-sim-100k-{z}.npy",
                  npy_size(TRIVIAQA_COUNTS[z], 769, 4), "triviaqa") for z in ZIPF]
    return items + [Item("drive", "sphere-ip-10m.json", "datasets/aker/sphere-ip-10m.json", None, "triviaqa")]


def sphere_items() -> list[Item]:
    return [Item("hf", "sphere-10m.npy", "datasets/aker/sphere-10m.npy", 61_440_000_128),
            Item("drive", "sphere-ip-10m.json", "datasets/aker/sphere-ip-10m.json", None, "triviaqa")]


MEDRAG_100K = {
    "medrag-ip_100k-requests.npy": "datasets/aker/medrag-ip_100k-requests.npy",
    "requests_100k.json": "datasets/medrag/requests_100k.json",
    "zipf_draw_100k.json": "datasets/medrag/zipf_draw_100k.json",
    "paraphrases_100k.json": "datasets/medrag/paraphrases_100k.json",
    "paraphrase_stats_100k.json": "datasets/medrag/paraphrase_stats_100k.json",
}
MEDRAG_SMALL = {"base.json": "datasets/medrag/base.json"}
MEDRAG_CORPUS = {
    "medrag-ip-docs.npy": "datasets/aker/medrag-ip-docs.npy",
    "base_pmids.npy": "datasets/medrag/base_pmids.npy",
    "medrag_pmids.npy": "datasets/medrag/medrag_pmids.npy",
}
MEDRAG_10K = {
    "medrag-ip-requests.npy": "datasets/aker/medrag-10k-ip-requests.npy",
    "requests.json": "datasets/medrag_10k/requests.json",
    "zipf_draw.json": "datasets/medrag_10k/zipf_draw.json",
    "paraphrases.json": "datasets/medrag_10k/paraphrases.json",
    "paraphrase_stats.json": "datasets/medrag_10k/paraphrase_stats.json",
}
TRIPCLICK = {
    "tripclick-ip-docs.npy": "datasets/aker/tripclick-ip-docs.npy",
    "tripclick-ip-requests-2018-01-01.npy": "datasets/aker/tripclick-ip-requests-2018-01-01.npy",
    "requests-2018-01-01.json": "datasets/tripclick/requests-2018-01-01.json",
}
SIZES = {"medrag-ip_100k-requests.npy": npy_size(100_000, 769, 4), "medrag-ip-docs.npy": npy_size(23_888_832, 769, 4),
         "tripclick-ip-docs.npy": npy_size(1_523_878, 769, 4),
         "tripclick-ip-requests-2018-01-01.npy": npy_size(100_000, 769, 4)}


def drive_items(files: dict, folder: str) -> list[Item]:
    return [Item("drive", name, dest, SIZES.get(name), folder) for name, dest in files.items()]


def medrag_items(corpus: bool = True, ten_k: bool = False) -> list[Item]:
    items = drive_items(MEDRAG_SMALL, "medrag") + drive_items(MEDRAG_100K, "medrag100k")
    if corpus:
        items += drive_items(MEDRAG_CORPUS, "medrag")
    if ten_k:
        items += drive_items(MEDRAG_10K, "medrag")
    return items


MODES = ("spacev", "triviaqa", "sphere", "medrag", "tripclick")


def items_for(mode: str, corpus: bool = True, ten_k: bool = False) -> list[Item]:
    if mode == "medrag":
        return medrag_items(corpus, ten_k)
    if mode == "tripclick":
        return drive_items(TRIPCLICK, "tripclick")
    return {"spacev": spacev_items, "triviaqa": triviaqa_items, "sphere": sphere_items}[mode]()


def find_entry(item: Item, listing: dict) -> dict | None:
    for entry in listing.get(item.folder, []):
        if entry["path"] == item.name:
            return entry
    return None


def esci_entries(listing: dict) -> list[dict]:
    out = []
    for entry in listing.get("esci", []):
        parts = entry["path"].split("/")
        if "ESCI" in parts[:-1]:
            rel = "/".join(parts[parts.index("ESCI") + 1:])
            out.append(dict(entry, destination=f"datasets/esci/{rel}"))
    return out


def action_for(dest: Path, expected: int | None, force: bool) -> str:
    if not dest.exists():
        return "download"
    if force:
        return "replace"
    if expected is not None and dest.resolve().stat().st_size != expected:
        return "replace"
    return "keep"


def needed_folders(modes: list[str], esci: bool, corpus: bool, ten_k: bool) -> list[str]:
    folders = {i.folder for m in modes for i in items_for(m, corpus, ten_k) if i.source == "drive"}
    return sorted(folders | ({"esci"} if esci else set()))


def build_actions(modes: list[str], root: Path, listing: dict, corpus: bool, force: bool, esci: bool,
                  ten_k: bool = False) -> list[dict]:
    actions = []
    for mode in modes:
        for item in items_for(mode, corpus, ten_k):
            entry, size = None, item.size
            if item.source == "drive":
                entry = find_entry(item, listing)
                if entry is None:
                    actions.append({"mode": mode, "item": item, "action": "missing", "size": size})
                    continue
                size = entry.get("size") or size
            actions.append({"mode": mode, "item": item, "entry": entry, "size": size,
                            "action": action_for(root / item.dest, size, force)})
    if esci:
        for entry in esci_entries(listing):
            item = Item("drive", Path(entry["destination"]).name, entry["destination"], None, "esci")
            actions.append({"mode": "esci", "item": item, "entry": entry, "size": entry.get("size"),
                            "action": action_for(root / item.dest, entry.get("size"), force)})
    return actions


def missing_message(actions: list[dict]) -> str:
    lines = [f"  {a['item'].name} is not in the {a['item'].folder} folder ({FOLDERS[a['item'].folder]})"
             for a in actions if a["action"] == "missing"]
    return "expected files are missing from their Drive folders, nothing was downloaded:\n" + "\n".join(lines)


def total_needed(actions: list[dict]) -> int:
    return sum(a["size"] or 0 for a in actions if a["action"] in ("download", "replace"))


def space_ok(need: int, free: int, floor_gb: float = MIN_FREE_GB) -> bool:
    return free - need >= floor_gb * 2**30


def list_drive(folder_id: str) -> list[dict]:
    import gdown
    files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True, use_cookies=False)
    return [{"path": f.path, "id": f.id, "size": getattr(f, "size", None)} for f in files]


def place(staged: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    shutil.move(str(staged), str(dest))


def download_item(action: dict, root: Path, staging: Path) -> None:
    item = action["item"]
    dest = root / item.dest
    if item.source == "hf":
        from workloads.q1.datasets import common
        dest.parent.mkdir(parents=True, exist_ok=True)
        common.download_file(f"{HF_BASE}/{item.name}", dest)
        return
    import gdown
    target = staging / Path(item.dest).name
    gdown.download(id=action["entry"]["id"], output=str(target), quiet=False)
    place(target, dest)


def print_actions(actions: list[dict]) -> None:
    for a in actions:
        item = a["item"]
        size = a["size"] if a["size"] is not None else "?"
        print(f"{a['action']:9s} {a['mode']:9s} {item.source:5s} {item.name} -> {item.dest} ({size} bytes)")


def print_listing() -> None:
    for mode in MODES:
        for item in items_for(mode, True, False):
            where = item.folder or "hf"
            print(f"{mode:9s} {where:11s} {item.name} -> {item.dest}")
    for name, dest in MEDRAG_10K.items():
        print(f"medrag    medrag      {name} -> {dest} (only with --medrag-10k)")
    print("esci      esci        every file under ESCI/ -> datasets/esci/ (--esci only)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for mode in MODES + ("esci", "all", "list"):
        parser.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--no-corpus", action="store_true")
    parser.add_argument("--medrag-10k", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--staging", type=Path, default=Path("datasets/.drive_staging"))
    parser.add_argument("--listing", type=Path, help="JSON {folder: [{path, id, size}]} instead of asking Drive")
    return parser.parse_args(argv)


def selected_modes(args: argparse.Namespace) -> list[str]:
    return list(MODES) if args.all else [m for m in MODES if getattr(args, m)]


def run_post_steps(modes: list[str], root: Path) -> None:
    """The inner-product SPHERE corpus is built from the downloaded base (cluster only)."""
    if "sphere" in modes:
        from scripts import build_sphere_ip
        build_sphere_ip.build_from_files(root / "datasets/aker/sphere-10m.npy",
                                         root / "datasets/aker/sphere-ip-10m.npy",
                                         root / "datasets/aker/sphere-ip-10m.json")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print_listing()
        return 0
    modes = selected_modes(args)
    if not modes and not args.esci:
        print("choose a dataset (--spacev --triviaqa --sphere --medrag --tripclick --esci) or --all")
        return 2
    corpus = not args.no_corpus
    folders = needed_folders(modes, args.esci, corpus, args.medrag_10k)
    if args.listing:
        listing = json.loads(args.listing.read_text())
    else:
        listing = {name: list_drive(FOLDERS[name]) for name in folders}
    actions = build_actions(modes, args.root, listing, corpus, args.force, args.esci, args.medrag_10k)
    print_actions(actions)
    if any(a["action"] == "missing" for a in actions):
        print(missing_message(actions))
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
            print(f"refusing {a['item'].name}: fewer than {MIN_FREE_GB} GB would remain free")
            return 2
        download_item(a, args.root, args.staging)
    shutil.rmtree(args.staging, ignore_errors=True)
    run_post_steps(modes, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
