#!/usr/bin/env python3
"""
Re-embed Aker's TriviaQA simZipf traces with the encoder of the SPHERE corpus.

Why: Aker's released query vectors (datasets/aker/triviaqa-sim-100k-<z>.npy)
were encoded with BAAI/bge-base-en-v1.5 (raw CLS output; the paper says
BGE-large), while the SPHERE base (sphere-10m.npy; its first 100K rows are
Hermes' SPHERE_100K with text) was encoded with the passage side of Sphere's
DPR_web bi-encoder (github.com/facebookresearch/Sphere, dpr_web_biencoder.cp;
re-encoding SPHERE_100K text as (title, text) pairs gives cosine 0.998 to the
stored vectors). So queries and data lived in different spaces. This keeps
Aker's database vectors and their request sequences, and replaces each request
vector by the DPR_web question-encoder embedding (CLS, max 256 tokens, uncased)
of the same question.

Steps:
  1. encode every line of triviaqa.txt with bge-base (to identify which line
     each released request vector is) and with the DPR_web question encoder;
  2. per trace, match each distinct released vector to its line (nearest
     bge-base embedding; reported: distance, ambiguity) and write
     triviaqa-dpr-sim-100k-<z>.npy (float32, same row order as the original)
     plus triviaqa-dpr-sim-100k-<z>.json (match statistics).
The DPR_web encoders converted to Hugging Face format live in
~/third_party/sphere/dpr_web_{question,ctx}_encoder (see the context doc).

  .venv/bin/python datasets/reembed_triviaqa_queries.py [--threads 24]
"""

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
AKER = Path("datasets/aker")
THIRD_PARTY = Path(os.environ.get("RECAL_THIRD_PARTY", Path.home() / "third_party"))
DPR_Q = THIRD_PARTY / "sphere" / "dpr_web_question_encoder"


def encode(model_path, texts, threads, batch=128):
    """CLS output of the last layer (both bge-base raw and DPR), in input order;
    batches sorted by length to limit padding."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_path)
    m = AutoModel.from_pretrained(model_path).eval()
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty((len(texts), m.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(texts), batch):
            idx = order[b:b + batch]
            t = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256,
                    return_tensors="pt")
            out[idx] = m(**t).last_hidden_state[:, 0].numpy()
            if (b // batch) % 200 == 0:
                logger.info(f"  {model_path}: {b + len(idx)} / {len(texts)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--zipf", nargs="+", default=["0.3", "0.6", "0.99"])
    args = ap.parse_args()
    lines = (AKER / "triviaqa.txt").read_text().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    uniq, inv = np.unique(np.array(lines, dtype=object), return_inverse=True)
    uniq = [str(u) for u in uniq]
    logger.info(f"{len(lines)} lines, {len(uniq)} distinct")
    cache = {}
    for name, path in (("bge", "BAAI/bge-base-en-v1.5"), ("dpr", str(DPR_Q))):
        f = AKER / f"triviaqa-lines-{name}.npy"
        if f.exists():
            cache[name] = np.load(f)
        else:
            logger.info(f"encoding {len(uniq)} distinct lines with {path}...")
            cache[name] = encode(path, uniq, args.threads)
            np.save(f, cache[name])
    E_bge, E_dpr = cache["bge"], cache["dpr"]
    sq = (E_bge.astype(np.float64) ** 2).sum(1).astype(np.float32)
    for z in args.zipf:
        trace = np.load(AKER / f"triviaqa-sim-100k-{z}.npy")
        vals, tinv = np.unique(trace, axis=0, return_inverse=True)
        tinv = tinv.reshape(-1)
        V = vals.astype(np.float32)
        best = np.empty(len(V), dtype=np.int64)
        d1 = np.empty(len(V))
        d2 = np.empty(len(V))
        for s in range(0, len(V), 512):           # 512 x lines float32 blocks (~1 GB)
            v = V[s:s + 512]
            d = sq[None, :] + (v * v).sum(1)[:, None] - 2.0 * (v @ E_bge.T)
            p = np.argpartition(d, 1, axis=1)[:, :2]
            dd = np.take_along_axis(d, p, axis=1)
            o = np.argsort(dd, axis=1)
            best[s:s + len(v)] = np.take_along_axis(p, o, axis=1)[:, 0]
            dd = np.sqrt(np.maximum(np.take_along_axis(dd, o, axis=1), 0))
            d1[s:s + len(v)], d2[s:s + len(v)] = dd[:, 0], dd[:, 1]
        # identical texts share an embedding, so the runner-up of a matched line can be at
        # distance 0 only if two distinct lines embed identically (counted as ambiguous)
        stats = {"zipf": z, "requests": int(len(trace)), "distinct_vectors": int(len(V)),
                 "match_l2_median": float(np.median(d1)), "match_l2_p99": float(np.percentile(d1, 99)),
                 "match_l2_max": float(d1.max()),
                 "share_match_l2_below_0.1": float(np.mean(d1 < 0.1)),
                 "share_runner_up_within_2x": float(np.mean(d2 < 2 * d1 + 1e-6)),
                 "distinct_lines_matched": int(len(np.unique(best)))}
        logger.info(json.dumps(stats))
        out = E_dpr[best][tinv]
        np.save(AKER / f"triviaqa-dpr-sim-100k-{z}.npy", out.astype(np.float32))
        (AKER / f"triviaqa-dpr-sim-100k-{z}.json").write_text(json.dumps(stats, indent=1))
        # per request: index into the sorted distinct lines of triviaqa.txt
        np.save(AKER / f"triviaqa-dpr-sim-100k-{z}-lineidx.npy", best[tinv].astype(np.int32))
        logger.info(f"wrote triviaqa-dpr-sim-100k-{z}.npy {out.shape}")


if __name__ == "__main__":
    main()
