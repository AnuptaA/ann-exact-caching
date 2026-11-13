#!/usr/bin/env python3
"""
PMIDs of MedRAG's PubMed corpus (Hugging Face dataset MedRAG/pubmed: 1166 JSONL files, 70 GB of
text, 23.9M snippets, one per PubMed article with an abstract; the corpus Proximity's MedRAG-Zipf
retrieves from). Streams every file and keeps only its PMIDs (the text is not stored), one
datasets/medrag/medrag_pmids/<file>.npy (int64) per file, written atomically; resumable; per-thread
rate limit (default 3 threads x 15 MB/s, the quiet-timing-guard budget of the other downloads);
retries on network errors (also after being suspended by a timing batch). Finally concatenates all
into datasets/medrag/medrag_pmids.npy.

  .venv/bin/python datasets/medrag/medrag_pmids.py
"""

import re
import time
import logging
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
OUT = Path("datasets/medrag/medrag_pmids")
BASE = "https://huggingface.co/datasets/MedRAG/pubmed/resolve/main/"
PMID = re.compile(rb'"PMID":\s*(\d+)')


def files():
    from huggingface_hub import HfApi
    return sorted(f.path for f in HfApi().list_repo_tree("MedRAG/pubmed", repo_type="dataset", recursive=True)
                  if f.path.endswith(".jsonl"))


def one(path, rate):
    dst = OUT / (Path(path).stem + ".npy")
    if dst.exists():
        return 0
    for attempt in range(50):
        try:
            ids, rest, t0, got = [], b"", time.time(), 0
            with urllib.request.urlopen(BASE + path, timeout=120) as r:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    got += len(buf)
                    lines = (rest + buf).split(b"\n")
                    rest = lines.pop()
                    for ln in lines:
                        m = PMID.findall(ln)
                        if m:
                            ids.append(int(m[-1]))
                    wait = got / rate - (time.time() - t0)
                    if wait > 0:
                        time.sleep(wait)
            if rest.strip():
                m = PMID.findall(rest)
                if m:
                    ids.append(int(m[-1]))
            tmp = dst.with_suffix(".tmp.npy")
            np.save(tmp, np.asarray(ids, np.int64))
            tmp.rename(dst)
            return len(ids)
        except Exception as e:                       # network error / suspended too long: retry
            logger.warning(f"{path}: {e!r}, retry {attempt + 1}")
            time.sleep(min(600, 10 * (attempt + 1)))
    raise RuntimeError(f"{path}: gave up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--rate-mb", type=float, default=15.0, help="MB/s per thread")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fs = files()
    logger.info(f"{len(fs)} files")
    done = 0
    with ThreadPoolExecutor(args.threads) as ex:
        for n in ex.map(lambda p: one(p, args.rate_mb * 1e6), fs):
            done += 1
            if done % 50 == 0:
                logger.info(f"{done} / {len(fs)} files")
    ids = np.concatenate([np.load(OUT / (Path(p).stem + ".npy")) for p in fs])
    u = np.unique(ids)
    np.save("datasets/medrag/medrag_pmids.npy", u)
    logger.info(f"{len(ids)} snippets, {len(u)} distinct PMIDs -> datasets/medrag/medrag_pmids.npy")


if __name__ == "__main__":
    main()
