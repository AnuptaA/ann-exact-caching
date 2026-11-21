"""Part B: bandwidth probe, estimate, then stratified slices and released query files.

usage: 06_partb_download.py OUT_DIR [N_BLOCKS] [BLOCK_ROWS]
Writes sphere_slice.npy (float32), sphere_ids.npy, spacev_slice.npy (float32),
spacev_ids.npy, aker_trivia_0.6.npy, aker_spacev_*.npy and triviaqa_head.txt.
"""

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from common import AKER, npy_header, range_get, read_stratified

out = Path(sys.argv[1])
n_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
block = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
probe = 50_000_000
range_get(AKER + "/sphere-10m.npy", 3_000_000_000, 3_000_000_000 + probe - 1)
rate = probe / (time.time() - t0)
print(f"probe: {probe} bytes in {time.time() - t0:.1f}s = {rate / 1e6:.1f} MB/s")
sizes = {
    "sphere slice": n_blocks * block * 768 * 8,
    "spacev slice": n_blocks * block * 100,
    "aker triviaqa 0.6 npy": 153_584_768,
    "aker spacev npy x3": 3 * 10_000_000,
    "triviaqa.txt head": 7_000_000,
}
total = sum(sizes.values())
print("estimate:", {k: f"{v / 1e6:.0f}MB" for k, v in sizes.items()})
print(f"estimate total: {total} bytes, {total / rate:.0f} s at probe rate")

hdr = npy_header(AKER + "/sphere-10m.npy")
x, ids = read_stratified(AKER + "/sphere-10m.npy", hdr["shape"][0], n_blocks, block, 768, "<f8", hdr["header"])
np.save(out / "sphere_slice.npy", x)
np.save(out / "sphere_ids.npy", ids)
print(f"sphere slice {x.shape} done at {time.time() - t0:.0f}s")

hdr = npy_header(AKER + "/spacev-10m.npy")
print("spacev header:", hdr)
dt = np.dtype(hdr["dtype"])
x, ids = read_stratified(AKER + "/spacev-10m.npy", hdr["shape"][0], n_blocks, block,
                         hdr["shape"][1], hdr["dtype"], hdr["header"])
np.save(out / "spacev_slice.npy", x)
np.save(out / "spacev_ids.npy", ids)
print(f"spacev slice {x.shape} {dt} done at {time.time() - t0:.0f}s")


def fetch(name: str, dest: str, end: int | None = None) -> None:
    request = urllib.request.Request(f"{AKER}/{name}", headers={"Range": f"bytes=0-{end}"} if end else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        (out / dest).write_bytes(response.read())


fetch("triviaqa-sim-100k-0.6.npy", "aker_trivia_0.6.npy")
for skew in ("0.3", "0.6", "0.99"):
    fetch(f"spacev-sim-100k-{skew}.npy", f"aker_spacev_{skew}.npy")
fetch("triviaqa.txt", "triviaqa_head.txt", 7_000_000)
print(f"all done at {time.time() - t0:.0f}s")
