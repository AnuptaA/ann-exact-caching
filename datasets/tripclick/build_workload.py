#!/usr/bin/env python3
"""
TripClick as a vector-caching workload (context doc 9x): a real health search log with natural
repetition, over the 1.52M MEDLINE documents of the TripClick benchmark.

Steps (each skipped when its output exists; run from the repository root with the .venv):
  1. searches: the click logs (datasets/tripclick/raw/logs/*.json, 5.3M clicks) in time order, one
     search per run of consecutive clicks in a session with the same normalized query (lowercase,
     stripped); empty queries dropped; then a window of --n consecutive searches from --start;
  2. documents: MedCPT article encoder (ncbi/MedCPT-Article-Encoder; [title, abstract] pairs,
     max 512 tokens, CLS), written in resumable shards; queries: MedCPT query encoder
     (ncbi/MedCPT-Query-Encoder, max 64 tokens, CLS). MedCPT scores by inner product, as the
     MedCPT PubMed embeddings MedRAG uses;
  3. inner product -> Euclidean (as datasets/make_sphere_ip_transform.py): every document gets
     sqrt(M^2 - |x|^2) appended (M = largest document norm), every query a 0, so Euclidean
     neighbors = inner-product neighbors (769-d float32).
Outputs (git-ignored; the TripClick terms forbid sharing any part of the data or derived data):
  datasets/aker/tripclick-ip-docs.npy (+ -f32 symlink), datasets/aker/tripclick-ip-requests-<start>.npy
  (one row per search, in order), datasets/tripclick/requests-<start>.json (texts, counts, window).

  .venv/bin/python datasets/tripclick/build_workload.py --start 2018-01-01 --n 100000
"""

import re
import json
import glob
import argparse
import logging
import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RAW = Path("datasets/tripclick/raw")
OUT = Path("datasets/tripclick")
AKER = Path("datasets/aker")


def searches():
    rows = []
    for f in sorted(glob.glob(str(RAW / "logs" / "*.json"))):
        for line in open(f, encoding="utf-8-sig"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = re.search(r"\d+", r.get("DateCreated") or "")
            if m:
                rows.append((int(m.group()), r.get("SessionId") or "", " ".join((r.get("Keywords") or "").lower().split())))
    rows.sort()
    out, last = [], {}
    for ms, s, k in rows:
        if not k or last.get(s) == k:
            continue
        last[s] = k
        out.append((ms, k))
    return out


_MODELS = {}


def encode(model_name, texts, max_len, batch, pairs=False, pad_multiple=None):
    """CLS embeddings, fp32 output. bf16 autocast on CPU (cosine to fp32 >= 0.99996 on a TripClick
    sample, ~2x faster) and length-sorted batches (little padding; order restored).
    pad_multiple: pad every batch to a multiple of this many tokens. Without it, length-sorted
    batches have hundreds of distinct shapes, and the CPU bf16 kernels keep per-shape state (most
    likely a repacked weight copy, ~220 MB for MedCPT): workers grew to ~36 GB each (11-13);
    64 leaves at most 8 shapes for 512 tokens."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if model_name not in _MODELS:
        _MODELS[model_name] = (AutoTokenizer.from_pretrained(model_name),
                               AutoModel.from_pretrained(model_name).eval())
    tok, model = _MODELS[model_name]
    if pairs:
        lens = tok([t for t, _ in texts], [a for _, a in texts], truncation=True, max_length=max_len)["input_ids"]
    else:
        lens = tok(list(texts), truncation=True, max_length=max_len)["input_ids"]
    order = np.argsort([len(x) for x in lens], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        for b in range(0, len(texts), batch):
            sel = order[b:b + batch]
            chunk = [texts[i] for i in sel]
            enc = tok(chunk, truncation=True, padding=True, return_tensors="pt", max_length=max_len,
                      pad_to_multiple_of=pad_multiple) if not pairs \
                else tok([t for t, _ in chunk], [a for _, a in chunk], truncation=True, padding=True,
                         return_tensors="pt", max_length=max_len, pad_to_multiple_of=pad_multiple)
            out[sel] = model(**enc).last_hidden_state[:, 0, :].float().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01", help="window start (UTC date)")
    ap.add_argument("--n", type=int, default=100000, help="consecutive searches in the window")
    ap.add_argument("--shard", type=int, default=20000, help="documents per embedding shard")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--part", default=None, help="i/N: only encode document shards s with s %% N == i "
                    "(parallel workers), write doc_shards/part_i_of_N.json, and stop")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    tag = args.start
    req_json = OUT / f"requests-{tag}.json"
    # 1. searches -> window
    if not req_json.exists() and args.part is None:
        s = searches()
        t0 = int(datetime.datetime.fromisoformat(args.start).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        i0 = next(i for i, (ms, _) in enumerate(s) if ms >= t0)
        win = s[i0:i0 + args.n]
        texts = sorted(set(k for _, k in win))
        idx = {k: i for i, k in enumerate(texts)}
        seq = [idx[k] for _, k in win]
        first, last = win[0][0], win[-1][0]
        info = {"start": args.start, "n": len(win), "distinct": len(texts),
                "from": datetime.datetime.utcfromtimestamp(first / 1000).isoformat(),
                "to": datetime.datetime.utcfromtimestamp(last / 1000).isoformat(),
                "repeat_share": 1 - len(texts) / len(win), "all_searches": len(s),
                "texts": texts, "seq": seq}
        req_json.write_text(json.dumps(info))
        logger.info(f"window: {len(win)} searches {info['from']} - {info['to']}, {len(texts)} distinct "
                    f"({100 * info['repeat_share']:.1f}% repeats)")
    info = json.loads(req_json.read_text()) if req_json.exists() else None
    # 2a. documents (shards)
    docs_npy = AKER / "tripclick-ip-docs.npy"
    shard_dir = OUT / "doc_shards"
    shard_dir.mkdir(exist_ok=True)
    if not docs_npy.exists():
        ids, pairs = [], []
        with open(RAW / "benchmark_tsv" / "documents" / "docs.tsv", encoding="utf-8") as f:
            for line in f:
                did, _, text = line.rstrip("\n").partition("\t")
                title, _, abstract = text.partition("<eot>")
                ids.append(did)
                pairs.append((title.strip(), abstract.strip()))
        logger.info(f"{len(pairs)} documents")
        pi, pn = (int(v) for v in args.part.split("/")) if args.part else (0, 1)
        for s0 in range(0, len(pairs), args.shard):
            f = shard_dir / f"docs_{s0 // args.shard:04d}.npy"
            if f.exists() or (s0 // args.shard) % pn != pi:
                continue
            x = encode("ncbi/MedCPT-Article-Encoder", pairs[s0:s0 + args.shard], 512, 32, pairs=True, pad_multiple=64)
            np.save(str(f) + ".tmp.npy", x)
            Path(str(f) + ".tmp.npy").rename(f)   # atomic: a killed worker leaves no partial shard
            logger.info(f"documents shard {s0 // args.shard} ({min(s0 + args.shard, len(pairs))} / {len(pairs)})")
        if args.part:
            (shard_dir / f"part_{pi}_of_{pn}.json").write_text(json.dumps({"part": args.part}))
            return
        x = np.concatenate([np.load(shard_dir / f"docs_{i:04d}.npy")
                            for i in range((len(pairs) + args.shard - 1) // args.shard)])
        # 3. inner product -> Euclidean
        norms = np.linalg.norm(x, axis=1)
        M = float(norms.max())
        x = np.hstack([x, np.sqrt(np.maximum(M * M - norms * norms, 0.0))[:, None]]).astype(np.float32)
        np.save(docs_npy, x)
        f32 = AKER / "tripclick-ip-docs-f32.npy"
        if not f32.exists():
            f32.symlink_to(docs_npy.name)
        (OUT / "docs.json").write_text(json.dumps({"n": len(ids), "M": M, "norm_min": float(norms.min()),
                                                   "norm_median": float(np.median(norms)), "ids": ids}))
        logger.info(f"wrote {docs_npy} {x.shape}, M {M:.3f}")
    if args.part:
        return
    # 2b. queries -> request vectors
    req_npy = AKER / f"tripclick-ip-requests-{tag}.npy"
    if not req_npy.exists():
        qv = encode("ncbi/MedCPT-Query-Encoder", info["texts"], 64, 256)
        qv = np.hstack([qv, np.zeros((len(qv), 1), np.float32)])
        np.save(req_npy, qv[np.asarray(info["seq"])].astype(np.float32))
        logger.info(f"wrote {req_npy} ({len(info['seq'])} requests)")


if __name__ == "__main__":
    main()
