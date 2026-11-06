#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Build an ESCI embedding subsample from the raw ESCI parquet files.

Follows datasets/esci/README_esci.md:
  - model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - product text: "title. first_bullet_point" (title alone if no bullets)
  - L2-normalized float32 embeddings (inner product == cosine)

Encodes a random subsample of products (the database) and all unique queries.
Requires torch + sentence-transformers (run with the repo's .venv python).

Outputs (in --out-dir):
  esci_db{N}_s{seed}.npy, esci_db{N}_s{seed}_product_ids.npy
  esci_queries.npy, esci_query_ids.npy, esci_query_texts.npy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def product_text(title, bullets) -> str:
    title = "" if title is None or pd.isna(title) else str(title).strip()
    if bullets is None or pd.isna(bullets):
        return title
    first = str(bullets).strip().split("\n")[0].strip()
    return f"{title}. {first}" if first else title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, default="datasets/esci/data_vectors/raw")
    ap.add_argument("--out-dir", type=str, default="datasets/esci/data_vectors")
    ap.add_argument("--n-db", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--skip-queries", action="store_true",
                    help="only encode products (queries already built)")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(args.threads)

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Reading products...", flush=True)
    products = pd.read_parquet(
        raw / "shopping_queries_dataset_products.parquet",
        columns=["product_id", "product_title", "product_bullet_point"])
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(products), size=args.n_db, replace=False))
    sub = products.iloc[idx]
    texts = [product_text(t, b) for t, b in
             zip(sub["product_title"], sub["product_bullet_point"])]

    print("Reading queries...", flush=True)
    examples = pd.read_parquet(raw / "shopping_queries_dataset_examples.parquet",
                               columns=["query_id", "query"])
    queries = examples.drop_duplicates("query_id").sort_values("query_id")
    print(f"  {len(products)} products, sampled {len(sub)}; "
          f"{len(queries)} unique queries", flush=True)

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

    def encode(strings):
        emb = model.encode(strings, batch_size=args.batch_size,
                           normalize_embeddings=True, show_progress_bar=True,
                           convert_to_numpy=True)
        return emb.astype(np.float32)

    print("Encoding products...", flush=True)
    db = encode(texts)
    tag = f"esci_db{args.n_db}_s{args.seed}"
    np.save(out / f"{tag}.npy", db)
    np.save(out / f"{tag}_product_ids.npy", sub["product_id"].to_numpy().astype(str))
    if args.skip_queries:
        print(f"Done: db {db.shape}", flush=True)
        return

    print("Encoding queries...", flush=True)
    q = encode(queries["query"].astype(str).tolist())
    np.save(out / "esci_queries.npy", q)
    np.save(out / "esci_query_ids.npy", queries["query_id"].to_numpy())
    np.save(out / "esci_query_texts.npy", queries["query"].to_numpy().astype(str))
    print(f"Done: db {db.shape}, queries {q.shape}", flush=True)


if __name__ == "__main__":
    main()
