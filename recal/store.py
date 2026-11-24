"""Object store for artifacts and results: S3 or a local directory.

Layout: artifacts/<dataset>/<artifact> (with artifacts/<dataset>/manifest.json) and
results/<question>/<run hash>/<file>. Every object has a <key>.sha256 sidecar (size and checksum) written after
the object, so an object without its sidecar counts as not uploaded. A transfer is skipped when the other side
already has the same size and checksum; pulls are verified.

RECAL_STORE is s3://bucket/prefix or a directory; RECAL_S3_BUCKET and RECAL_S3_PREFIX work as well. S3 credentials
come from boto3's default chain (the instance role).
"""

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SIDECAR = ".sha256"
WORKERS = 8


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class LocalBackend:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put_file(self, key: str, path: Path) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        shutil.copyfile(path, tmp)
        tmp.replace(dest)

    def get_file(self, key: str, path: Path) -> None:
        shutil.copyfile(self._path(key), path)

    def put_bytes(self, key: str, data: bytes) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def get_bytes(self, key: str) -> bytes | None:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if base.is_file():
            return [prefix]
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*")
                      if p.is_file() and not p.name.endswith(".part"))


class S3Backend:
    def __init__(self, bucket: str, prefix: str = ""):
        import boto3
        self.client = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, key: str, path: Path) -> None:
        self.client.upload_file(str(path), self.bucket, self._key(key))

    def get_file(self, key: str, path: Path) -> None:
        self.client.download_file(self.bucket, self._key(key), str(path))

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def keys(self, prefix: str) -> list[str]:
        out, strip = [], len(self.prefix) + 1 if self.prefix else 0
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket,
                                                                          Prefix=self._key(prefix)):
            out += [o["Key"][strip:] for o in page.get("Contents", [])]
        return sorted(out)


def open_store(url: str | None = None):
    url = url or os.environ.get("RECAL_STORE")
    if url is None and os.environ.get("RECAL_S3_BUCKET"):
        return Store(S3Backend(os.environ["RECAL_S3_BUCKET"], os.environ.get("RECAL_S3_PREFIX", "")))
    if url is None:
        raise SystemExit("set RECAL_STORE (s3://bucket/prefix or a directory) or RECAL_S3_BUCKET")
    if url.startswith("s3://"):
        bucket, _, prefix = url[5:].partition("/")
        return Store(S3Backend(bucket, prefix))
    return Store(LocalBackend(url))


class Store:
    def __init__(self, backend, workers: int = WORKERS):
        self.backend, self.workers = backend, workers

    def _remote_sum(self, key: str) -> dict | None:
        raw = self.backend.get_bytes(key + SIDECAR)
        return json.loads(raw) if raw else None

    def push_file(self, key: str, path: Path) -> bool:
        """Upload one file; False when the store already has it."""
        path = Path(path)
        info = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        if self._remote_sum(key) == info:
            return False
        self.backend.put_file(key, path)
        self.backend.put_bytes(key + SIDECAR, json.dumps(info).encode())
        return True

    def pull_file(self, key: str, path: Path) -> bool:
        """Download one file and verify it; False when the local copy already matches."""
        path = Path(path)
        info = self._remote_sum(key)
        if info is None:
            raise FileNotFoundError(f"{key} is not in the store (or has no checksum)")
        if path.exists() and path.stat().st_size == info["size"] and sha256_file(path) == info["sha256"]:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f".{path.name}.part")
        self.backend.get_file(key, part)
        if part.stat().st_size != info["size"] or sha256_file(part) != info["sha256"]:
            part.unlink(missing_ok=True)
            raise ChecksumError(f"{key}: downloaded file does not match its checksum")
        part.replace(path)
        return True

    def _each(self, fn, items) -> list:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(fn, items))

    def push_tree(self, directory: Path, prefix: str) -> int:
        directory = Path(directory)
        files = [p for p in sorted(directory.rglob("*")) if p.is_file() and not p.name.startswith(".")]
        done = self._each(lambda p: self.push_file(f"{prefix}/{p.relative_to(directory)}", p), files)
        return sum(done)

    def keys(self, prefix: str) -> list[str]:
        return [k for k in self.backend.keys(prefix) if not k.endswith(SIDECAR)]

    def pull_prefix(self, prefix: str, directory: Path) -> int:
        directory, start = Path(directory), len(prefix.rstrip("/")) + 1
        keys = self.keys(prefix)
        done = self._each(lambda k: self.pull_file(k, directory / k[start:] if k != prefix else directory), keys)
        return sum(done)

    def ls(self, prefix: str = "") -> list[dict]:
        out = []
        for k in self.keys(prefix):
            info = self._remote_sum(k) or {}
            out.append({"key": k, "size": info.get("size"), "sha256": info.get("sha256")})
        return out


def artifact_key(dataset: str, name: str) -> str:
    return f"artifacts/{dataset}/{name}"


def result_prefix(question: str, run_hash: str) -> str:
    return f"results/{question}/{run_hash}"
