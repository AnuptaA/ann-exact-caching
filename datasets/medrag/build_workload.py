#!/usr/bin/env python3
"""
MedRAG-Zipf as a vector-caching workload (context doc 9y), following Proximity (arXiv 2503.05530,
4.2): the 500 PubMedQA questions of MIRAGE (datasets/medrag/benchmark.json), 10,000 independent
draws with Zipf 0.8 (no temporal locality), every occurrence reworded (paraphrase.py), retrieval
over MedRAG's PubMed corpus (23.9M articles) with MedCPT (inner product).

  draw     ranks = a random permutation of the 500 questions, P(rank k) ~ k^-0.8, 10,000 draws;
           re-drawn (next seed) until every question appears at least once, as in the paper
           -> datasets/medrag/zipf_draw.json
  base     NCBI's MedCPT article embeddings (pubmed_medcpt/, download_medcpt.sh) restricted to the
           MedRAG PMIDs (medrag_pmids.py), PMID order, inner product -> Euclidean (sqrt(M^2 - |x|^2)
           appended, M = largest norm) -> datasets/aker/medrag-ip-docs.npy (N x 769 float32, written
           through a memory map) + -f32 symlink, datasets/medrag/base_pmids.npy, base.json
  queries  the rewordings, in stream order, with the MedCPT query encoder (64 tokens, CLS), 0 appended
           -> datasets/aker/medrag-ip-requests.npy, datasets/medrag/requests.json (texts, question ids)

  .venv/bin/python datasets/medrag/build_workload.py draw|base|queries
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
D = Path("datasets/medrag")
AKER = Path("datasets/aker")
EMB = D / "pubmed_medcpt"


def draw(args):
    b = json.loads((D / "benchmark.json").read_text())["pubmedqa"]
    keys = sorted(b)
    qs = [b[k]["question"] for k in keys]
    seed = args.seed
    while True:
        rng = np.random.default_rng(seed)
        rank_of = rng.permutation(len(qs))                  # question i has rank rank_of[i] + 1
        p = (rank_of + 1.0) ** -args.zipf
        p /= p.sum()
        seq = rng.choice(len(qs), size=args.n, p=p)
        counts = np.bincount(seq, minlength=len(qs))
        if counts.min() >= 1:
            break
        seed += 1
    info = {"zipf": args.zipf, "n": args.n, "seed": seed, "pubmedqa_ids": keys, "questions": qs,
            "counts": counts.tolist(), "seq": seq.tolist(), "max_count": int(counts.max())}
    (D / "zipf_draw.json").write_text(json.dumps(info))
    logger.info(f"seed {seed}: {args.n} draws over {len(qs)} questions, max count {counts.max()}, "
                f"median {int(np.median(counts))}, min {counts.min()}")


def chunks():
    i = 0
    while (EMB / f"embeds_chunk_{i}.npy.ok").exists():
        yield i
        i += 1


def base(args):
    want = np.load(D / "medrag_pmids.npy")
    sel, tot = [], 0
    for i in chunks():
        pm = np.asarray(json.loads((EMB / f"pmids_chunk_{i}.json").read_text()), np.int64)
        tot += len(pm)
        keep = np.nonzero(np.isin(pm, want))[0]
        sel.append((i, keep, pm[keep]))
    n = sum(len(k) for _, k, _ in sel)
    pmids = np.concatenate([p for _, _, p in sel])
    logger.info(f"{tot} NCBI articles in {len(sel)} chunks; {n} of the {len(want)} MedRAG PMIDs have an embedding")
    norms = np.empty(n, np.float32)
    out = AKER / "medrag-ip-docs.npy"
    tmp = AKER / "medrag-ip-docs.tmp.npy"
    y = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, 769))
    o = 0
    # one SEQUENTIAL read per chunk (11-14: memory-mapped fancy indexing faulted 4 KB pages at
    # ~10 MB/s on the busy EBS volume); rows and norms written, the appended column after M is known
    for i, keep, _ in sel:
        x = np.load(EMB / f"embeds_chunk_{i}.npy")[keep].astype(np.float32, copy=False)
        y[o:o + len(x), :768] = x
        norms[o:o + len(x)] = np.linalg.norm(x, axis=1)
        o += len(x)
        del x
        logger.info(f"chunk {i}: {o} / {n}")
    M = float(norms.max())
    for s0 in range(0, n, 1 << 22):
        nb = norms[s0:s0 + (1 << 22)].astype(np.float64)
        y[s0:s0 + len(nb), 768] = np.sqrt(np.maximum(M * M - nb * nb, 0.0))
    y.flush()
    del y
    tmp.rename(out)
    f32 = AKER / "medrag-ip-docs-f32.npy"
    if not f32.exists():
        f32.symlink_to(out.name)
    np.save(D / "base_pmids.npy", pmids)
    (D / "base.json").write_text(json.dumps({"n": n, "medrag_pmids": int(len(want)), "ncbi_articles": tot,
                                             "M": M, "norm_min": float(norms.min()),
                                             "norm_median": float(np.median(norms))}))
    logger.info(f"wrote {out} ({n} x 769), M {M:.3f}")


def queries(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location("tripclick_build", Path(__file__).parent.parent / "tripclick" / "build_workload.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    encode = tc.encode                                          # MedCPT, bf16, sorted batches
    import torch
    torch.set_num_threads(args.threads)
    draw_ = json.loads((D / "zipf_draw.json").read_text())
    par = {int(k): v for k, v in json.loads((D / "paraphrases.json").read_text()).items()}
    rng = np.random.default_rng(draw_["seed"] + 1)
    for i in par:                                               # rewordings in random order per question
        rng.shuffle(par[i])
    nxt = {i: 0 for i in par}
    texts = []
    for q in draw_["seq"]:
        texts.append(par[q][nxt[q]])
        nxt[q] += 1
    assert len(set(texts)) == len(texts), "rewordings must be unique"
    v = encode("ncbi/MedCPT-Query-Encoder", texts, 64, 256)
    v = np.hstack([v, np.zeros((len(v), 1), np.float32)]).astype(np.float32)
    np.save(AKER / "medrag-ip-requests.npy", v)
    (D / "requests.json").write_text(json.dumps({"texts": texts, "question": draw_["seq"]}))
    # how close are rewordings of one question compared with different questions (MedCPT space)
    u = v[:, :768] / np.linalg.norm(v[:, :768], axis=1, keepdims=True)
    q = np.asarray(draw_["seq"])
    s = rng.choice(len(v), size=min(2000, len(v)), replace=False)
    sims = u[s] @ u[s].T
    same = q[s][:, None] == q[s][None, :]
    np.fill_diagonal(same, False)
    off = ~same
    np.fill_diagonal(off, False)
    logger.info(f"wrote {len(v)} requests; cosine same question {sims[same].mean():.3f}, "
                f"different questions {sims[off].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["draw", "base", "queries"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--zipf", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    {"draw": draw, "base": base, "queries": queries}[args.step](args)


if __name__ == "__main__":
    main()
