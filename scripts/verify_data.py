#!/usr/bin/env python3
"""Checks of the TripClick and MedRAG vector files and the data manifest.

  python scripts/verify_data.py tripclick [--dir datasets/aker]
  python scripts/verify_data.py medrag [--dir datasets/medrag] [--aker-dir datasets/aker]
  python scripts/verify_data.py manifest [--out datasets/MANIFEST.md]

Large arrays are memory-mapped and sampled. Nothing is downloaded or modified.
"""

import json
import hashlib
import argparse
from pathlib import Path

import numpy as np

SAMPLE = 2000
NORM_TOL = 1e-4

ZIPF = ("0.3", "0.6", "0.99")
AKER_EXPECTED = {
    "spacev-10m.npy": ((10_000_000, 100), "int8", "cluster only"),
    "sphere-ip-10m.npy": ((10_000_000, 769), "float32", "cluster only"),
    **{f"spacev-sim-100k-{z}.npy": ((100_000, 100), "int8", "local") for z in ZIPF},
    **{f"triviaqa-ip-sim-100k-{z}.npy": ((100_000, 769), "float32", "local")
       for z in ZIPF},
}

MEDRAG_EXPECTED = {
    "medrag-ip-docs.npy": ((23_888_832, 769), "float32", "cluster only"),
    "base_pmids.npy": ((23_888_832,), "int64", "cluster only"),
    "medrag_pmids.npy": ((23_895_135,), "int64", "cluster only"),
    "medrag-ip-requests.npy": ((10_000, 769), "float32", "local"),
}


def sample_rows(array: np.ndarray, count: int = SAMPLE, seed: int = 0) -> np.ndarray:
    count = min(count, len(array))
    index = np.sort(np.random.default_rng(seed).choice(len(array), count, replace=False))
    return np.asarray(array[index]).astype(np.float64)


def check_ip_transform(docs: np.ndarray, requests: np.ndarray, norm_max: float | None = None) -> dict:
    """Documents share one norm (M) after the appended coordinate; requests end in 0."""
    problems = []
    if docs.shape[1] != requests.shape[1]:
        problems.append(f"dimension mismatch {docs.shape[1]} vs {requests.shape[1]}")
    rows = sample_rows(docs)
    if np.isnan(rows).any():
        problems.append("NaN in a document sample")
    norms = np.linalg.norm(rows, axis=1)
    spread = float((norms.max() - norms.min()) / norms.max())
    if spread > NORM_TOL:
        problems.append(f"document norms differ (relative spread {spread:.2e})")
    if norm_max is not None and abs(norms.max() - norm_max) > 1e-3 * norm_max:
        problems.append(f"document norm {norms.max():.5f} differs from M {norm_max:.5f}")
    queries = sample_rows(requests, len(requests))
    if np.isnan(queries).any():
        problems.append("NaN in requests")
    if np.abs(queries[:, -1]).max() != 0.0:
        problems.append("last coordinate of a request is not 0")
    return {"norm": float(norms.max()), "spread": spread, "problems": problems}


def check_zipf_draw(draw: dict, n_requests: int = 10_000, n_questions: int = 500) -> list[str]:
    problems = []
    seq = np.asarray(draw["seq"])
    if len(seq) != n_requests:
        problems.append(f"draw has {len(seq)} requests, expected {n_requests}")
    counts = np.bincount(seq, minlength=n_questions)
    if len(counts) != n_questions or counts.min() < 1:
        problems.append("not every question appears at least once")
    if list(counts) != list(draw.get("counts", counts)):
        problems.append("counts do not match the sequence")
    return problems


def check_shape(path: Path, shape: tuple, dtype: str) -> list[str]:
    array = np.load(path, mmap_mode="r")
    if array.shape != shape or str(array.dtype) != dtype:
        return [f"{path.name} is {array.shape} {array.dtype}, expected {shape} {dtype}"]
    rows = sample_rows(array, 200) if array.dtype.kind == "f" else array[:200]
    if array.dtype.kind == "f" and np.isnan(rows).any():
        return [f"NaN in a sample of {path.name}"]
    return []


def verify_spacev(directory: Path) -> int:
    problems = []
    for name, (shape, dtype, _) in AKER_EXPECTED.items():
        if name.startswith("spacev") and (directory / name).exists():
            problems += check_shape(directory / name, shape, dtype)
    return report(problems)


def verify_sphere(directory: Path) -> int:
    path = directory / "sphere-ip-10m.npy"
    if not path.exists():
        return report([f"missing {path}"])
    shape, dtype, _ = AKER_EXPECTED[path.name]
    problems = check_shape(path, shape, dtype)
    meta = directory / "sphere-ip-10m.json"
    norm_max = json.loads(meta.read_text()).get("M") if meta.exists() else None
    streams = [directory / f"triviaqa-ip-sim-100k-{z}.npy" for z in ZIPF]
    queries = next((s for s in streams if s.exists()), None)
    if not problems and queries is not None:
        docs = np.load(path, mmap_mode="r")
        problems += check_ip_transform(docs, np.load(queries, mmap_mode="r"),
                                       norm_max)["problems"]
    return report(problems)


def verify_triviaqa(directory: Path) -> int:
    problems = []
    for z in ZIPF:
        path = directory / f"triviaqa-ip-sim-100k-{z}.npy"
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        shape, dtype, _ = AKER_EXPECTED[path.name]
        problems += check_shape(path, shape, dtype)
        queries = sample_rows(np.load(path, mmap_mode="r"), SAMPLE)
        if np.abs(queries[:, -1]).max() != 0.0:
            problems.append(f"last coordinate of a request in {path.name} is not 0")
    return report(problems)


def report(problems: list[str]) -> int:
    for problem in problems:
        print("PROBLEM:", problem)
    return 1 if problems else 0


def verify_tripclick(directory: Path) -> int:
    docs = np.load(directory / "tripclick-ip-docs.npy", mmap_mode="r")
    requests = np.load(directory / "tripclick-ip-requests-2018-01-01.npy", mmap_mode="r")
    result = check_ip_transform(docs, requests)
    print(f"tripclick docs {docs.shape} {docs.dtype}, requests {requests.shape} {requests.dtype}, "
          f"document norm {result['norm']:.5f}")
    for problem in result["problems"]:
        print("PROBLEM:", problem)
    return 1 if result["problems"] else 0


def verify_medrag(directory: Path, aker_dir: Path) -> int:
    failures = 0
    for name, (shape, dtype, where) in MEDRAG_EXPECTED.items():
        path = (aker_dir if name.startswith("medrag-ip") else directory) / name
        if not path.exists():
            print(f"missing ({where}): {name}")
            continue
        array = np.load(path, mmap_mode="r")
        if array.shape != shape or str(array.dtype) != dtype:
            print(f"PROBLEM: {name} is {array.shape} {array.dtype}, expected {shape} {dtype}")
            failures += 1
    draw_path = directory / "zipf_draw.json"
    if draw_path.exists():
        for problem in check_zipf_draw(json.loads(draw_path.read_text())):
            print("PROBLEM:", problem)
            failures += 1
    base_path = directory / "base.json"
    docs_path = aker_dir / "medrag-ip-docs.npy"
    requests_path = aker_dir / "medrag-ip-requests.npy"
    if base_path.exists() and requests_path.exists():
        base = json.loads(base_path.read_text())
        requests = np.load(requests_path, mmap_mode="r")
        if np.abs(sample_rows(requests, len(requests))[:, -1]).max() != 0.0:
            print("PROBLEM: last coordinate of a request is not 0")
            failures += 1
        if docs_path.exists():
            docs = np.load(docs_path, mmap_mode="r")
            for problem in check_ip_transform(docs, requests, base["M"])["problems"]:
                print("PROBLEM:", problem)
                failures += 1
    return 1 if failures else 0


def sha256(path: Path, block: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_rows(paths: list[Path]) -> list[str]:
    rows = []
    for path in paths:
        size = path.stat().st_size
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r")
            shape, dtype = "x".join(map(str, array.shape)), str(array.dtype)
        else:
            shape, dtype = "-", "-"
        rows.append(f"| `{path.as_posix()}` | {size} | {shape} | {dtype} | `{sha256(path)}` | on disk |")
    return rows


def write_manifest(out: Path, paths: list[Path]) -> None:
    lines = [
        "# Data manifest",
        "",
        "Data files are gitignored. Sizes in bytes. TripClick's terms forbid sharing the data or anything derived "
        "from it.",
        "",
        "| File | Bytes | Shape | Dtype | SHA-256 | Status |",
        "|---|---|---|---|---|---|",
        *manifest_rows([p for p in paths if p.exists()]),
    ]
    on_disk = {p.name for p in paths if p.exists()}
    for name, (shape, dtype, where) in {**AKER_EXPECTED, **MEDRAG_EXPECTED}.items():
        if name not in on_disk:
            status = "expected, cluster only" if where == "cluster only" else "expected"
            lines.append(f"| `{name}` | - | {'x'.join(map(str, shape))} | {dtype} | - | {status} |")
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tripclick = sub.add_parser("tripclick")
    tripclick.add_argument("--dir", type=Path, default=Path("datasets/aker"))
    medrag = sub.add_parser("medrag")
    medrag.add_argument("--dir", type=Path, default=Path("datasets/medrag"))
    medrag.add_argument("--aker-dir", type=Path, default=Path("datasets/aker"))
    for name in ("spacev", "sphere", "triviaqa"):
        sub.add_parser(name).add_argument("--dir", type=Path, default=Path("datasets/aker"))
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--out", type=Path, default=Path("datasets/MANIFEST.md"))
    manifest.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    if args.command == "tripclick":
        return verify_tripclick(args.dir)
    if args.command == "medrag":
        return verify_medrag(args.dir, args.aker_dir)
    if args.command in ("spacev", "sphere", "triviaqa"):
        return {"spacev": verify_spacev, "sphere": verify_sphere,
                "triviaqa": verify_triviaqa}[args.command](args.dir)
    write_manifest(args.out, args.paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
