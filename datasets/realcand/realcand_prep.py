#!/usr/bin/env python3
"""Candidate real workloads (doc 9av): request statistics that predict whether our cache would help, before
building full workloads. Two candidates, each a real timestamped query log with a ~10M-vector corpus to search:

  wildchat     WildChat-1M (allenai/WildChat-1M): first user turn of each English conversation, by timestamp;
               corpus SPHERE-10M (Wikipedia passages, Sphere DPR_web; datasets/aker/sphere-10m-f32.npy, inner
               product), queries with the DPR_web question encoder (as the TriviaQA re-embedding)
  diffusiondb  DiffusionDB 2M+14M (poloclub/diffusiondb, metadata-large.parquet): one request per prompt
               submission (rows with the same user, prompt and timestamp are the images of one submission), by
               timestamp; corpus LAION-10M (9.6M CLIP ViT-B/32 image embeddings, gdrive:Datasets/LAION),
               queries with the CLIP ViT-B/32 text encoder (text -> image search, inner product)

Steps (each writes datasets/realcand/<name>/...):
  requests <name>   all requests (text, time, user) -> requests_all.json; a window of --n consecutive requests
                    from --start -> requests.json
  encode <name>     query vectors of the window -> q.npy
  search <name>     exact top-10 over the corpus (FAISS flat, inner product) -> top10.npy   [.venv-ann]
  stats <name>      the 9am statistics (exact repeats, repeat gaps, near repeats = >= 5 of the top-10 shared with
                    an earlier request, nearest earlier query) after a 20k warm-up -> stats.json
"""
import json
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
D = Path("datasets/realcand")
SPHERE = Path("datasets/aker/sphere-10m-f32.npy")
LAION_DIR = D / "laion10m"


def norm_text(s):
    return " ".join(str(s).lower().split())


def requests(a):
    out = D / a.name
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    if a.name == "wildchat":
        import pyarrow.parquet as pq
        p = snapshot_download("allenai/WildChat-1M", repo_type="dataset", allow_patterns=["data/*.parquet"])
        rows = []
        for f in sorted(Path(p, "data").glob("*.parquet")):
            t = pq.read_table(f, columns=["conversation", "timestamp", "language", "hashed_ip"]).to_pylist()
            for r in t:
                if r["language"] != "English" or not r["conversation"]:
                    continue
                first = next((m["content"] for m in r["conversation"] if m.get("role") == "user"), None)
                if first and first.strip():
                    rows.append((r["timestamp"].timestamp(), first.strip(), r["hashed_ip"] or ""))
            logger.info(f"{f.name}: {len(rows)} English first turns so far")
    else:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("poloclub/diffusiondb", "metadata-large.parquet", repo_type="dataset")
        df = pd.read_parquet(f, columns=["prompt", "user_name", "timestamp"])
        df = df.dropna(subset=["prompt"])
        df = df[df["prompt"].str.strip() != ""]
        sub = df.groupby(["user_name", "prompt", "timestamp"], sort=False).size().reset_index()   # one submission
        rows = [(ts.timestamp(), p.strip(), u) for u, p, ts in zip(sub["user_name"], sub["prompt"], sub["timestamp"])]
        logger.info(f"{len(df)} images -> {len(rows)} prompt submissions")
    rows.sort(key=lambda r: r[0])
    allf = {"t": [r[0] for r in rows], "text": [r[1] for r in rows], "user": [r[2] for r in rows]}
    (out / "requests_all.json").write_text(json.dumps(allf))
    s = a.start if a.start >= 0 else max(0, len(rows) // 2 - a.n // 2)        # default: the middle of the log
    w = rows[s:s + a.n]
    (out / "requests.json").write_text(json.dumps({"start": s, "t": [r[0] for r in w], "text": [r[1] for r in w],
                                                   "user": [r[2] for r in w]}))
    span = (rows[-1][0] - rows[0][0]) / 86400 if rows else 0
    logger.info(f"{a.name}: {len(rows)} requests over {span:.0f} days; window {s}..{s + len(w)} "
                f"({(w[-1][0] - w[0][0]) / 3600:.1f} h)")


def encode(a):
    import torch
    torch.set_num_threads(a.threads)
    r = json.loads((D / a.name / "requests.json").read_text())
    texts = r["text"]
    order = np.argsort([len(t) for t in texts], kind="stable")
    with torch.no_grad():
        if a.name == "wildchat":
            from transformers import AutoModel, AutoTokenizer
            path = str(Path.home() / "third_party/sphere/dpr_web_question_encoder")
            tok, m = AutoTokenizer.from_pretrained(path), AutoModel.from_pretrained(path).eval()
            out = np.empty((len(texts), m.config.hidden_size), np.float32)
            for b in range(0, len(texts), 128):
                idx = order[b:b + 128]
                t = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256, return_tensors="pt")
                out[idx] = m(**t).last_hidden_state[:, 0].numpy()
                if (b // 128) % 100 == 0:
                    logger.info(f"encoded {b + len(idx)} / {len(texts)}")
        else:
            from transformers import CLIPModel, CLIPTokenizer
            name = "openai/clip-vit-base-patch32"
            tok, m = CLIPTokenizer.from_pretrained(name), CLIPModel.from_pretrained(name).eval()
            out = np.empty((len(texts), m.config.projection_dim), np.float32)
            for b in range(0, len(texts), 256):
                idx = order[b:b + 256]
                t = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=77, return_tensors="pt")
                e = m.get_text_features(**t)
                out[idx] = torch.nn.functional.normalize(e, dim=-1).numpy()
                if (b // 256) % 100 == 0:
                    logger.info(f"encoded {b + len(idx)} / {len(texts)}")
    np.save(D / a.name / "q.npy", out)
    logger.info(f"wrote {D / a.name / 'q.npy'} {out.shape}")


def corpus_chunks(name):
    """(offset, vectors) chunks of the corpus, float32."""
    if name == "wildchat":
        X = np.load(SPHERE, mmap_mode="r")
        for s in range(0, len(X), 1_000_000):
            yield s, np.ascontiguousarray(X[s:s + 1_000_000], dtype=np.float32)
    else:
        for f in sorted((LAION_DIR / "data_vectors").glob("shard_*.npz")):
            z = np.load(f)
            ids = z["passage_ids"]
            yield ids, np.ascontiguousarray(z["embeddings"], dtype=np.float32)


def search(a):
    import faiss
    faiss.omp_set_num_threads(a.threads)
    q = np.ascontiguousarray(np.load(D / a.name / "q.npy"), dtype=np.float32)
    k = 10
    best_d = np.full((len(q), k), -np.inf, np.float32)
    best_i = np.full((len(q), k), -1, np.int64)
    for off, X in corpus_chunks(a.name):
        ix = faiss.IndexFlatIP(X.shape[1])
        ix.add(X)
        d, i = ix.search(q, k)
        gid = (i + off) if np.isscalar(off) or isinstance(off, int) else np.asarray(off)[i]
        cd = np.hstack([best_d, d])
        ci = np.hstack([best_i, gid])
        o = np.argsort(-cd, axis=1)[:, :k]
        best_d, best_i = np.take_along_axis(cd, o, 1), np.take_along_axis(ci, o, 1)
        logger.info(f"chunk {off if np.isscalar(off) else int(np.asarray(off)[0])}: done")
    np.save(D / a.name / "top10.npy", best_i)
    np.save(D / a.name / "top10_ip.npy", best_d)
    logger.info(f"wrote {D / a.name / 'top10.npy'}")


def stats(a):
    r = json.loads((D / a.name / "requests.json").read_text())
    texts, users = r["text"], r["user"]
    top = np.load(D / a.name / "top10.npy")
    q = np.load(D / a.name / "q.npy")
    n, W = len(texts), min(20000, len(texts) // 5)
    keys = [norm_text(t) for t in texts]
    last, gaps, rep, rep_user = {}, [], np.zeros(n, bool), np.zeros(n, bool)
    last_user = {}
    for i, kk in enumerate(keys):
        if kk in last:
            rep[i] = True
            gaps.append(i - last[kk]) if i >= W else None
            rep_user[i] = last_user[kk] == users[i]
        last[kk], last_user[kk] = i, users[i]
    post = np.arange(n) >= W
    # near repeats: a first-time (not exact-repeat) request whose top-10 shares >= 5 ids with an earlier request's
    from collections import defaultdict, Counter
    post_ids = defaultdict(list)
    best = np.zeros(n, np.int32)
    for i in range(n):
        c = Counter()
        for v in top[i]:
            for j in post_ids[int(v)][-200:]:
                c[j] += 1
        best[i] = max(c.values()) if c else 0
        for v in top[i]:
            post_ids[int(v)].append(i)
    first = post & ~rep
    # nearest earlier query (cosine), on a sample
    rng = np.random.default_rng(0)
    smp = rng.choice(np.flatnonzero(post), size=min(3000, post.sum()), replace=False)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    near_cos = np.array([float((qn[:i] @ qn[i]).max()) for i in smp])
    g = np.array(gaps)
    users_post = len(set(u for u, p in zip(users, post) if p))
    out = {"workload": a.name, "requests": n, "warmup": W, "window_hours": (r["t"][-1] - r["t"][0]) / 3600,
           "distinct_requests_share": len(set(keys)) / n, "users_in_window": users_post,
           "exact_repeat_share": float(rep[post].mean()), "exact_repeat_same_user_share": float(rep_user[post].mean()),
           "repeat_gap_median": float(np.median(g)) if len(g) else None,
           "repeat_within_100": float((g <= 100).sum() / post.sum()) if len(g) else 0.0,
           "repeat_within_1000": float((g <= 1000).sum() / post.sum()) if len(g) else 0.0,
           "near5_share_of_all": float((first & (best >= 5)).sum() / post.sum()),
           "near5_share_of_first_time": float((best[first] >= 5).mean()),
           "near8_share_of_all": float((first & (best >= 8)).sum() / post.sum()),
           "nearest_earlier_cosine_median": float(np.median(near_cos)),
           "nearest_earlier_cosine_p90": float(np.percentile(near_cos, 90)),
           "corpus": "SPHERE-10M (10M Wikipedia passages, 768-d DPR_web)" if a.name == "wildchat" else
                     "LAION-10M (9.6M images, 512-d CLIP ViT-B/32)"}
    allr = json.loads((D / a.name / "requests_all.json").read_text())
    out["log_requests_total"] = len(allr["t"])
    out["log_days"] = (allr["t"][-1] - allr["t"][0]) / 86400
    (D / a.name / "stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["requests", "encode", "search", "stats"])
    ap.add_argument("name", choices=["wildchat", "diffusiondb"])
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--start", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    {"requests": requests, "encode": encode, "search": search, "stats": stats}[a.step](a)


if __name__ == "__main__":
    main()
