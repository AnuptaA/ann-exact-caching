"""Resumable HTTP downloads (with byte ranges) and big-ann style .bin helpers."""

import base64
import hashlib
import http.client
import json
import logging
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_BYTES = 64 * 1024 * 1024
TIMEOUT = 60
RETRIES = 5


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return int(response.headers["Content-Length"])


class RangeNotSupported(OSError):
    """The server ignored the Range header; retrying cannot help."""


def fetch_range(url: str, start: int, length: int) -> bytes:
    """Bytes [start, start + length) of url; fails unless the server honors ranges."""
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if response.status != 206:
            raise RangeNotSupported(f"{url} ignored the Range header (status {response.status})")
        data = response.read()
    if len(data) != length:
        raise OSError(f"{url}: asked for {length} bytes at {start}, got {len(data)}")
    return data


def fetch_range_retry(url: str, start: int, length: int, tries: int = RETRIES,
                      sleep=time.sleep) -> bytes:
    """fetch_range with exponential backoff; every failure is logged, the last one is raised."""
    for attempt in range(1, tries + 1):
        try:
            return fetch_range(url, start, length)
        except RangeNotSupported:
            raise
        except (OSError, http.client.HTTPException) as exc:
            logger.warning(f"{url} bytes {start}+{length}: attempt {attempt}/{tries} failed: {exc!r}")
            if attempt == tries:
                raise OSError(f"{url}: giving up after {tries} attempts at byte {start}") from exc
            sleep(min(2 ** attempt, 60))


def download_slice(
    url: str,
    dest: Path,
    byte_start: int,
    byte_end: int,
    header: bytes = b"",
    chunk_bytes: int = CHUNK_BYTES,
) -> Path:
    """Write header + url[byte_start:byte_end] to dest, resuming a partial file.

    The file appears under its final name only when complete and of the expected size.
    Progress is logged about every 5% with the transfer rate.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    if byte_end <= byte_start:
        raise ValueError("empty byte range")
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = len(header) + byte_end - byte_start
    if part.exists() and part.stat().st_size > total:
        part.unlink()
    if not part.exists():
        part.write_bytes(header)
    done = max(part.stat().st_size, len(header))
    start_done, started, last_pct = done, time.time(), -1
    with part.open("r+b") as f:
        f.seek(done)
        while done < total:
            take = min(chunk_bytes, total - done)
            f.write(fetch_range_retry(url, byte_start + done - len(header), take))
            done += take
            pct = int(100 * done / total) // 5 * 5
            if pct > last_pct:
                last_pct = pct
                rate = (done - start_done) / max(time.time() - started, 1e-6) / 2 ** 20
                logger.info(f"{dest.name}: {done / total:.0%} of {total / 2 ** 30:.2f} GiB at {rate:.0f} MiB/s")
    if part.stat().st_size != total:
        raise OSError(f"{part}: {part.stat().st_size} bytes written, expected {total}")
    part.rename(dest)
    return dest


def download_file(url: str, dest: Path, chunk_bytes: int = CHUNK_BYTES, verify: bool = True) -> Path:
    """Resumable download, then checked against the published checksum (or recorded on first use)."""
    dest = download_slice(url, dest, 0, content_length(url), chunk_bytes=chunk_bytes)
    if verify:
        ensure_verified(dest, url)
    return dest


# -- integrity ---------------------------------------------------------------

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX32 = re.compile(r"^[0-9a-f]{32}$")
S3_MULTIPART = re.compile(r"^[0-9a-f]{32}-(\d+)$")
LFS_POINTER = re.compile(r"oid sha256:([0-9a-f]{64})\nsize (\d+)")
S3_HOSTS = ("dl.fbaipublicfiles.com",)
PART_SIZES_MIB = (8, 16, 5, 32, 64, 100, 10, 15, 20, 25, 50, 128)
HASH_BLOCK = 8 * 2 ** 20


class ChecksumError(OSError):
    """A downloaded file does not match what its host publishes, or changed since first use."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def head_headers(url: str, follow: bool = True, tries: int = 3, sleep=time.sleep):
    """Response headers of a HEAD request; follow=False returns the redirect's own headers."""
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(_NoRedirect)
    for attempt in range(1, tries + 1):
        try:
            with opener.open(urllib.request.Request(url, method="HEAD"), timeout=TIMEOUT) as response:
                return response.headers
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400 and not follow:
                return exc.headers
            error = exc
        except OSError as exc:
            error = exc
        logger.warning(f"HEAD {url}: attempt {attempt}/{tries} failed: {error!r}")
        if attempt == tries:
            raise OSError(f"HEAD {url} failed {tries} times") from error
        sleep(2 ** attempt)


def parse_lfs_pointer(text: str) -> tuple[str, int] | None:
    match = LFS_POINTER.search(text)
    return (match.group(1), int(match.group(2))) if match else None


def checksum_from_headers(host: str, headers) -> tuple[str, str, int | None] | None:
    """(kind, expected, size) from response headers; None when the host publishes nothing usable."""
    size = headers.get("x-linked-size") or headers.get("Content-Length")
    size = int(size) if size and size.isdigit() else None
    if host == "huggingface.co":
        etag = (headers.get("x-linked-etag") or "").strip('"')
        size = int(headers["x-linked-size"]) if headers.get("x-linked-size") else size
        if HEX64.match(etag):
            return "sha256", etag, size
        if HEX40.match(etag):
            return "git-sha1", etag, size
        return None
    if headers.get("Content-MD5"):
        return "md5-b64", headers["Content-MD5"], size
    etag = (headers.get("ETag") or "").strip('"')
    if host in S3_HOSTS and (HEX32.match(etag) or S3_MULTIPART.match(etag)):
        return ("md5", etag, size) if HEX32.match(etag) else ("s3-multipart", etag, size)
    return None


def published_checksum(url: str) -> tuple[str, str, int | None] | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc == "github.com" and "/raw/" in parsed.path:
        pointer_url = f"https://raw.githubusercontent.com{parsed.path.replace('/raw/', '/', 1)}"
        with urllib.request.urlopen(pointer_url, timeout=TIMEOUT) as response:
            pointer = parse_lfs_pointer(response.read(4096).decode("utf-8", "ignore"))
        return ("sha256", pointer[0], pointer[1]) if pointer else None
    headers = head_headers(url, follow=parsed.netloc != "huggingface.co")
    return checksum_from_headers(parsed.netloc, headers)


def _hash_file(path: Path, make_hash, prefix: bytes = b""):
    digest = make_hash()
    digest.update(prefix)
    size = Path(path).stat().st_size
    done, last_pct = 0, -1
    with Path(path).open("rb") as f:
        while block := f.read(HASH_BLOCK):
            digest.update(block)
            done += len(block)
            if size > 2 ** 30 and done * 10 // size > last_pct:
                last_pct = done * 10 // size
                logger.info(f"{Path(path).name}: hashed {done / size:.0%}")
    return digest


def s3_multipart_etag(path: Path, parts: int) -> str | None:
    """Composite ETag for the part size that gives `parts` parts (None when no usual size does)."""
    size = Path(path).stat().st_size
    for mib in PART_SIZES_MIB:
        part = mib * 2 ** 20
        if -(-size // part) != parts:
            continue
        digests = []
        with Path(path).open("rb") as f:
            while block := f.read(part):
                digests.append(hashlib.md5(block).digest())
        return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{parts}"
    return None


def digest_of(path: Path, kind: str, expected: str = "") -> str:
    """The file's digest in the same format a host publishes for `kind`."""
    size = Path(path).stat().st_size
    if kind in ("sha256", "sha256-tofu"):
        return _hash_file(path, hashlib.sha256).hexdigest()
    if kind == "md5":
        return _hash_file(path, hashlib.md5).hexdigest()
    if kind == "md5-b64":
        return base64.b64encode(_hash_file(path, hashlib.md5).digest()).decode()
    if kind == "git-sha1":
        return _hash_file(path, hashlib.sha1, f"blob {size}\0".encode()).hexdigest()
    if kind == "s3-multipart":
        return s3_multipart_etag(path, int(S3_MULTIPART.match(expected).group(1))) or "no matching part size"
    raise ValueError(f"unknown checksum kind {kind}")


def ensure_verified(dest: Path, url: str, use_published: bool = True) -> dict:
    """Verify dest against the host's checksum, else record its sha256 on first use.

    A sidecar <dest>.verified.json (size, mtime) lets reruns skip hashing. A file that
    fails is renamed to <dest>.corrupt so the next run downloads it again.
    """
    dest = Path(dest)
    sidecar = dest.with_name(dest.name + ".verified.json")
    stat = dest.stat()
    recorded = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if recorded and (recorded["size"], recorded["mtime_ns"]) == (stat.st_size, stat.st_mtime_ns):
        return recorded
    published = published_checksum(url) if use_published else None
    if published is not None:
        kind, expected, size = published
        actual = digest_of(dest, kind, expected)
    else:
        kind, size = "sha256-tofu", None
        actual = digest_of(dest, kind)
        expected = recorded["digest"] if recorded and recorded["kind"] == kind else actual
    if actual != expected or (size is not None and size != stat.st_size):
        corrupt = dest.with_name(dest.name + ".corrupt")
        dest.replace(corrupt)
        sidecar.unlink(missing_ok=True)
        raise ChecksumError(f"{dest.name}: {kind} {actual} (size {stat.st_size}), expected {expected}"
                            f"{'' if size is None else f' (size {size})'}; moved to {corrupt.name}")
    record = {"url": url, "kind": kind, "digest": actual, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(sidecar)
    logger.info(f"{dest.name}: {kind} {actual[:16]}... " +
                ("verified against the published value" if published else "recorded (no published checksum)"))
    return record


def bin_header(n: int, d: int) -> bytes:
    return struct.pack("<ii", n, d)


def read_bin_header(path: Path) -> tuple[int, int]:
    with Path(path).open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"{path}: shorter than the 8-byte header")
    return struct.unpack("<ii", raw)


def memmap_bin(path: Path, dtype) -> np.ndarray:
    """Memory-map a big-ann .bin file (int32 n, int32 d header, then rows)."""
    n, d = read_bin_header(path)
    expected = 8 + n * d * np.dtype(dtype).itemsize
    size = Path(path).stat().st_size
    if n < 0 or d < 0 or size != expected:
        raise ValueError(f"{path}: {size} bytes, header (n={n}, d={d}) says {expected}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def check_free_disk(path: Path, needed_bytes: int) -> None:
    import shutil
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < needed_bytes:
        raise OSError(f"{path}: {free / 2**30:.1f} GiB free, need {needed_bytes / 2**30:.1f} GiB")
