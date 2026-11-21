"""Page cache control for timing runs (Linux). Elsewhere eviction and residency report 'unsupported' / 'unknown'."""
import ctypes
import importlib.util
import mmap
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

_CHUNK = 1 << 24


def _supported() -> bool:
    return sys.platform.startswith("linux") and hasattr(os, "posix_fadvise")


def expand_files(paths: Iterable) -> List[Path]:
    """Regular files in paths; directories are walked recursively."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file()))
        elif p.is_file():
            files.append(p)
    return files


def evict(paths: Iterable) -> dict:
    """Drop clean cached pages of the files (POSIX_FADV_DONTNEED). Does not fail where unsupported."""
    if not _supported():
        return {"state": "unsupported", "files": 0}
    files = expand_files(paths)
    errors = 0
    for f in files:
        try:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            errors += 1
    return {"state": "evicted" if errors == 0 else "partial", "files": len(files), "errors": errors}


def _file_residency(path: Path, libc) -> tuple:
    size = path.stat().st_size
    if size == 0:
        return 0, 0
    page = mmap.PAGESIZE
    pages = (size + page - 1) // page
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            vec = (ctypes.c_ubyte * pages)()
            libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            resident = int(np.frombuffer(vec, dtype=np.uint8).__and__(1).sum())
        finally:
            mm.close()
    return resident, pages


def residency(paths: Iterable) -> dict:
    """Fraction of the files' pages that are resident in the page cache (mincore)."""
    unknown = {"state": "unknown", "fraction": None}
    if not sys.platform.startswith("linux"):
        return unknown
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        resident = total = 0
        for f in expand_files(paths):
            r, t = _file_residency(f, libc)
            resident += r
            total += t
    except (OSError, ValueError, AttributeError):
        return unknown
    return {"state": "ok", "fraction": resident / total if total else None, "pages": total, "resident": resident}


def shuffled_arms(arms: Sequence, seed: int, repetition: int) -> list:
    """Arm order for one repetition: a seeded permutation, different per repetition."""
    order = np.random.default_rng([int(seed), int(repetition)]).permutation(len(arms))
    return [arms[i] for i in order]


def record(state_before: dict, state_after: dict) -> dict:
    return {"before": state_before, "after": state_after}


def diskann_uses_direct_io() -> Optional[bool]:
    """True if the installed diskannpy binaries contain the string O_DIRECT, False if not, None if diskannpy
    is not importable. A missing string does not prove buffered I/O (the flag is usually a compiled-in constant).
    Needs a cluster check: also compare a cold and warm run, or strace the open() calls of one search."""
    spec = importlib.util.find_spec("diskannpy")
    if spec is None or not spec.submodule_search_locations:
        return None
    needle = b"O_DIRECT"
    for root in map(Path, spec.submodule_search_locations):
        for so in root.rglob("*.so*"):
            tail = b""
            with open(so, "rb") as fh:
                while True:
                    chunk = fh.read(_CHUNK)
                    if not chunk:
                        break
                    if needle in tail + chunk:
                        return True
                    tail = chunk[-len(needle):]
    return False
