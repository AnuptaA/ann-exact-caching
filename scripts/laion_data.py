#!/usr/bin/env python3
"""LAION-prompts: verify the fetched files and build the float32 corpus file from the float16 shards.

  python -m scripts.laion_data verify [--dir datasets/laion]
  python -m scripts.laion_data build [--dir datasets/laion] [--out datasets/aker/laion-ip-docs.npy]
  python -m scripts.laion_data check-truth [--prep results/online/laion-prompts]

Corpus row r is the r-th vector of shard_000.npz ... shard_099.npz concatenated in order. Shards are read one at a
time and written into a memory-mapped output, so the corpus is never held in memory.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import data_fetch
from scripts.verify_data import sha256

DIM = 512
TOP_DEPTH = 128
NORM_TOL = 1e-2
SAMPLE = 2000
ID_KEYS = ("passage_ids", "passage_id", "ids")
VECTOR_KEYS = ("embeddings", "vectors", "embedding", "data")


def shard_paths(directory: Path) -> list[Path]:
    return [directory / "shards" / data_fetch.laion_shard_name(i) for i in range(data_fetch.LAION_SHARDS)]


def pick_key(names: list[str], preferred: tuple[str, ...], what: str, path: Path) -> str:
    for key in preferred:
        if key in names:
            return key
    raise KeyError(f"{path.name} has no {what} array (keys: {names})")


def shard_ids(path: Path) -> np.ndarray:
    with np.load(path) as shard:
        return shard[pick_key(list(shard.files), ID_KEYS, "passage id", path)]


def shard_vectors(path: Path) -> np.ndarray:
    with np.load(path) as shard:
        names = [n for n in shard.files if n not in ID_KEYS]
        return shard[pick_key(names, VECTOR_KEYS, "vector", path)]


def shard_rows(paths: list[Path]) -> list[int]:
    return [len(shard_ids(p)) for p in paths]


def check_requests(directory: Path) -> list[str]:
    problems = []
    queries = np.load(directory / "q.npy", mmap_mode="r")
    requests = np.load(directory / "rc-diffusiondb-requests.npy", mmap_mode="r")
    stream = np.load(directory / "streams.npz")["trace"]
    top = np.load(directory / "top128.npy", mmap_mode="r")
    if requests.shape != (data_fetch.LAION_REQUESTS, DIM):
        problems.append(f"requests are {requests.shape}, expected {(data_fetch.LAION_REQUESTS, DIM)}")
    if queries.shape != (data_fetch.LAION_QUERIES, DIM):
        problems.append(f"q.npy is {queries.shape}, expected {(data_fetch.LAION_QUERIES, DIM)}")
    if top.shape != (len(queries), TOP_DEPTH):
        problems.append(f"top128.npy is {top.shape}, expected {(len(queries), TOP_DEPTH)}")
    if len(stream) != len(requests):
        problems.append(f"stream has {len(stream)} entries, requests {len(requests)}")
    if len(stream) and (stream.min() < 0 or stream.max() >= len(queries)):
        problems.append(f"stream indices outside [0, {len(queries)})")
    if problems:
        return problems
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(requests), min(SAMPLE, len(requests)), replace=False))
    if not np.array_equal(np.asarray(requests[sample]), np.asarray(queries[stream[sample]])):
        problems.append("request vectors differ from q.npy rows at the stream indices")
    norms = np.linalg.norm(np.asarray(queries[np.sort(rng.choice(len(queries), min(SAMPLE, len(queries)), replace=False))]), axis=1)
    if np.abs(norms - 1).max() > 1e-4:
        problems.append(f"query norms are not 1 (max deviation {np.abs(norms - 1).max():.2e})")
    return problems


def check_corpus(directory: Path) -> list[str]:
    paths = shard_paths(directory)
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        return [f"missing {len(missing)} shards, first {missing[0]}"]
    problems = []
    rows = shard_rows(paths)
    if sum(rows) != data_fetch.LAION_ROWS:
        problems.append(f"shards hold {sum(rows)} rows, expected {data_fetch.LAION_ROWS}")
    mapping = np.load(directory / "row_to_passage_id.npy", mmap_mode="r")
    if len(mapping) != sum(rows):
        return problems + [f"row_to_passage_id has {len(mapping)} entries, shards hold {sum(rows)}"]
    start = 0
    for path, count in zip(paths, rows):
        if not np.array_equal(shard_ids(path), np.asarray(mapping[start:start + count])):
            problems.append(f"{path.name}: passage ids differ from row_to_passage_id rows {start}..{start + count}")
        start += count
    for path in (paths[0], paths[-1]):
        vectors = shard_vectors(path)
        norms = np.linalg.norm(vectors[:SAMPLE].astype(np.float32), axis=1)
        if vectors.shape[1] != DIM or np.abs(norms - 1).max() > NORM_TOL:
            problems.append(f"{path.name}: dimension {vectors.shape[1]} or norms off (max deviation "
                            f"{np.abs(norms - 1).max():.2e})")
    return problems


def check_top128_rows(directory: Path) -> list[str]:
    top = np.load(directory / "top128.npy", mmap_mode="r")
    high = int(np.asarray(top[:5000]).max())
    return [f"top128 row id {high} is outside the corpus"] if high >= data_fetch.LAION_ROWS else []


def verify(directory: Path) -> int:
    needed = [name for name in data_fetch.LAION_PROMPTS if name != "README.md"]
    missing = [name for name in needed if not (directory / name).exists()]
    if missing:
        print("PROBLEM: missing " + ", ".join(str(directory / m) for m in missing))
        return 1
    problems = check_requests(directory) + check_top128_rows(directory)
    problems += check_corpus(directory) if (directory / "shards").exists() else ["no shards (fetched with --no-corpus)"]
    for problem in problems:
        print("PROBLEM:", problem)
    if not problems:
        print(f"laion: {data_fetch.LAION_ROWS} corpus rows, {data_fetch.LAION_REQUESTS} requests, "
              f"{data_fetch.LAION_QUERIES} distinct queries, ids and norms consistent")
    return 1 if problems else 0


def build_base(directory: Path, out: Path) -> dict:
    paths = shard_paths(directory)
    rows = shard_rows(paths)
    total = sum(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    target = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(total, DIM))
    start = 0
    for path, count in zip(paths, rows):
        vectors = shard_vectors(path)
        if vectors.shape != (count, DIM):
            raise ValueError(f"{path.name}: vectors are {vectors.shape}, expected {(count, DIM)}")
        target[start:start + count] = vectors.astype(np.float32)
        start += count
    target.flush()
    del target
    tmp.replace(out)
    info = {"rows": total, "dim": DIM, "dtype": "float32", "shards": len(paths), "sha256": sha256(out)}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def link_inputs(directory: Path, base: Path) -> None:
    """Names the workload registry reads: the request file, and the float32 copy that loaders memory-map."""
    requests = base.with_name("laion-requests.npy")
    f32 = base.with_name(base.stem + "-f32.npy")
    for link, target in ((requests, (directory / "rc-diffusiondb-requests.npy").resolve()), (f32, base.resolve())):
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


def check_truth(prep: Path, directory: Path, count: int = 200, min_overlap: float = 0.98) -> list[str]:
    """Compare the prep's exact top-500 with the shipped top-128 for a sample of queries (matched by vector)."""
    shipped = np.load(directory / "q.npy", mmap_mode="r")
    shipped_top = np.load(directory / "top128.npy", mmap_mode="r")
    queries = np.load(prep / "q.npy", mmap_mode="r")
    top = np.load(prep / "top500.npy", mmap_mode="r")
    row_of = {np.asarray(shipped[i]).tobytes(): i for i in range(len(shipped))}
    sample = np.sort(np.random.default_rng(0).choice(len(queries), min(count, len(queries)), replace=False))
    overlaps = []
    for i in sample:
        j = row_of.get(np.asarray(queries[i], dtype=np.float32).tobytes())
        if j is None:
            return [f"prep query {i} is not in the shipped q.npy"]
        mine, theirs = set(np.asarray(top[i][:TOP_DEPTH]).tolist()), set(np.asarray(shipped_top[j]).tolist())
        overlaps.append(len(mine & theirs) / TOP_DEPTH)
    mean = float(np.mean(overlaps))
    print(f"top-128 spot check: {len(sample)} queries, mean overlap {mean:.4f}, minimum {min(overlaps):.3f}")
    return [] if min(overlaps) >= min_overlap else [f"top-128 overlap below {min_overlap} for some queries"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("verify", "build", "check-truth"))
    parser.add_argument("--dir", type=Path, default=Path("datasets/laion"))
    parser.add_argument("--out", type=Path, default=Path("datasets/aker/laion-ip-docs.npy"))
    parser.add_argument("--prep", type=Path, default=Path("results/online/laion-prompts"))
    args = parser.parse_args(argv)
    if args.command == "verify":
        return verify(args.dir)
    if args.command == "check-truth":
        problems = check_truth(args.prep, args.dir)
        for problem in problems:
            print("PROBLEM:", problem)
        return 1 if problems else 0
    existing = args.out.with_suffix(".json")
    if args.out.exists() and existing.exists():
        info = json.loads(existing.read_text())
        if info["rows"] == data_fetch.LAION_ROWS:
            print(f"{args.out} already built ({info['rows']} rows, sha256 {info['sha256'][:12]})")
            link_inputs(args.dir, args.out)
            return 0
    print(build_base(args.dir, args.out))
    link_inputs(args.dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
