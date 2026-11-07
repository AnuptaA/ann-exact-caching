"""Resumable HTTP downloads (with byte ranges) and big-ann style .bin helpers."""

import http.client
import logging
import struct
import time
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


def download_file(url: str, dest: Path, chunk_bytes: int = CHUNK_BYTES) -> Path:
    return download_slice(url, dest, 0, content_length(url), chunk_bytes=chunk_bytes)


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
