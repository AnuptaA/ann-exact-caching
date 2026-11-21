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
