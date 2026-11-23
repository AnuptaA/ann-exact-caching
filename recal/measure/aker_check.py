"""Startup check of the Aker query filter: the HNSW parameters FAISS actually holds must be the paper's."""

import os
import re
import tempfile

PAPER_FILTER = {"faiss_M": 4, "efConstruction": 8, "efSearch": 16}
FILTER_LINE = re.compile(r"^FILTER\s+(.*)$", re.M)


class CaptureStderr:
    """Collects what C++ code writes to file descriptor 2 inside the block (std::cerr of the library)."""

    def __enter__(self):
        self.file = tempfile.TemporaryFile(mode="w+b")
        self.saved = os.dup(2)
        os.dup2(self.file.fileno(), 2)
        self.text = ""
        return self

    def __exit__(self, *exc):
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.file.seek(0)
        self.text = self.file.read().decode("utf-8", "replace")
        self.file.close()
        if self.text:
            os.write(2, self.text.encode())
        return False


def parse_filter(text: str) -> dict | None:
    """Key=value pairs of the first FILTER line, ints where possible; None when there is no such line."""
    m = FILTER_LINE.search(text)
    if m is None:
        return None
    out = {}
    for item in m.group(1).split():
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


def check_filter(held: dict, expected: dict = PAPER_FILTER) -> None:
    for key, want in expected.items():
        if held.get(key) != want:
            raise SystemExit(f"Aker filter holds {key}={held.get(key)}, expected {want} "
                             "(build the shim against the vendored library: AKER_SRC=vendored sh aker_real/build.sh)")


def verify_filter(text: str, expected: dict = PAPER_FILTER, allow_unverified: bool = False) -> dict | None:
    """The FILTER line in `text` must show `expected`. A missing line or other values fail the run unless
    allow_unverified (the unmodified library prints no line and hard-codes efConstruction 16, efSearch 8)."""
    held = parse_filter(text)
    if held is None:
        if allow_unverified:
            return None
        raise SystemExit("no FILTER line at creation: the filter parameters cannot be verified "
                         "(build with AKER_SRC=vendored, or pass --allow-unverified-filter)")
    try:
        check_filter(held, expected)
    except SystemExit:
        if allow_unverified:
            return held
        raise
    return held


def library_fingerprint(repo: str = ".") -> dict:
    """Repo commit and the git tree hash of the vendored Aker library, for every result."""
    import subprocess

    def git(*args):
        try:
            out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"code_commit": git("rev-parse", "HEAD"), "aker_tree": git("rev-parse", "HEAD:third_party/aker"),
            "source": "vendored", "filter": PAPER_FILTER}
