"""ESCI: the full product set and distinct query texts, same 384-d model as before.

Products are embedded in catalog order from "title. first_bullet"; embeddings are
L2-normalized float32, as in datasets/esci/README_esci.md. There is no official
query set, so the official queries are the distinct query texts.
"""

import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from workloads.q1 import atomic

logger = logging.getLogger(__name__)

RAW_BASE = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"
RAW_FILES = ("shopping_queries_dataset_products.parquet", "shopping_queries_dataset_examples.parquet")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_PRODUCTS = 1_814_924
EXPECTED_QUERY_IDS = 130_652
COUNT_TOLERANCE = 0.03
DIM = 384
SHARD_TEXTS = 100_000
TEXT_RECIPE = "product_title + '. ' + first line of product_bullet_point (product_text)"


def product_text(title, bullets) -> str:
    title = "" if title is None or _isna(title) else str(title).strip()
    if bullets is None or _isna(bullets):
        return title
    first = str(bullets).strip().split("\n")[0].strip()
    return f"{title}. {first}" if first else title


def _isna(value) -> bool:
    return isinstance(value, float) and np.isnan(value)


def product_texts(products) -> list[str]:
    """Texts in catalog order from a frame with product_title, product_bullet_point."""
    return [product_text(t, b) for t, b in
            zip(products["product_title"], products["product_bullet_point"])]


def distinct_queries(examples) -> tuple[list[str], np.ndarray]:
    """Distinct query texts (first occurrence by query_id) and their query ids."""
    by_id = examples.drop_duplicates("query_id").sort_values("query_id")
    by_text = by_id.drop_duplicates("query")
    return by_text["query"].astype(str).tolist(), by_text["query_id"].to_numpy()


def check_counts(n_products: int, n_query_ids: int, n_query_texts: int,
                 expected_products: int = EXPECTED_PRODUCTS,
                 expected_query_ids: int = EXPECTED_QUERY_IDS,
                 tolerance: float = COUNT_TOLERANCE) -> dict:
    """Log and return the counts; raise when far from the expected sizes."""
    counts = {"products": n_products, "query_ids": n_query_ids, "query_texts": n_query_texts}
    logger.info(f"ESCI counts {counts} (expected products ~{expected_products}, "
                f"query ids ~{expected_query_ids})")
    for name, got, want in (("products", n_products, expected_products),
                            ("query ids", n_query_ids, expected_query_ids)):
        if abs(got - want) > tolerance * want:
            raise ValueError(f"ESCI {name}: {got}, expected about {want}")
    return counts


def make_encoder(batch_size: int = 256, device: str = "cpu"):
    """Encoder returning L2-normalized float32 embeddings (needs sentence-transformers)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device=device)

    def encode(texts):
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
        return emb.astype(np.float32)
    return encode


def verify_embeddings_match(embeddings, texts, encode, n: int = 64, min_cos: float = 0.999,
                            seed: int = 0) -> float:
    """Re-encode n random rows; the stored rows must match (cosine) to be reused."""
    if len(embeddings) != len(texts):
        raise ValueError(f"{len(embeddings)} embeddings for {len(texts)} texts")
    idx = np.sort(np.random.default_rng(seed).choice(len(texts), min(n, len(texts)), replace=False))
    fresh = encode([texts[i] for i in idx])
    stored = np.asarray(embeddings[idx], dtype=np.float32)
    cos = (fresh * stored).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(stored, axis=1))
    if cos.min() < min_cos:
        raise ValueError(f"stored embeddings do not match the recipe (min cosine {cos.min():.4f})")
    return float(cos.min())


def recipe(dim: int = DIM, shard_texts: int = SHARD_TEXTS) -> dict:
    return {"model": MODEL, "text": TEXT_RECIPE, "normalize": True, "dim": dim, "shard_texts": shard_texts}


def texts_sha256(texts) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def code_commit() -> str:
    repo = Path(__file__).resolve().parents[3]
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _check_shard(vectors: np.ndarray, n: int, dim: int, name: str) -> None:
    if vectors.shape != (n, dim) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError(f"{name}: shape {vectors.shape} dtype {vectors.dtype}, expected ({n}, {dim}) float32")
    if np.abs(np.linalg.norm(vectors, axis=1) - 1.0).max() > 1e-3:
        raise ValueError(f"{name}: rows are not L2-normalized")


def embed_sharded(texts: list[str], encode, out_path: Path, dim: int = DIM,
                  shard_texts: int = SHARD_TEXTS) -> Path:
    """Embed texts into out_path in resumable shards, with a manifest of how they were made.

    An existing out_path is reused only if its manifest matches these texts and recipe, or
    (no manifest) if a sampled re-encode matches; otherwise this raises instead of overwriting.
    """
    out_path = Path(out_path)
    manifest_path = out_path.with_name(out_path.name + ".manifest.json")
    shard_dir = out_path.with_name(out_path.name + ".shards")
    want = {"recipe": recipe(dim, shard_texts), "n": len(texts), "texts_sha256": texts_sha256(texts)}
    if out_path.exists():
        have = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if have is not None and {k: have.get(k) for k in want} != want:
            raise ValueError(f"{out_path} was made from other texts or another recipe: {have}")
        embeddings = np.load(out_path, mmap_mode="r")
        min_cos = verify_embeddings_match(embeddings, texts, encode)
        if have is None:
            atomic.write_json(manifest_path, {**want, "adopted": True, "min_cosine": min_cos,
                                              "code_commit": code_commit()}, indent=1)
        logger.info(f"{out_path.name}: reusing existing embeddings (sample min cosine {min_cos:.5f})")
        return out_path
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest = shard_dir / "manifest.json"
    if shard_manifest.exists() and json.loads(shard_manifest.read_text()) != want:
        raise ValueError(f"{shard_dir} holds shards of other texts or another recipe; remove it to start over")
    atomic.write_json(shard_manifest, want)
    n_shards = -(-len(texts) // shard_texts)
    began, built = time.time(), 0
    for i in range(n_shards):
        path = shard_dir / f"shard_{i:05d}.npy"
        if path.exists():
            continue
        chunk = texts[i * shard_texts:(i + 1) * shard_texts]
        vectors = np.asarray(encode(chunk), dtype=np.float32)
        _check_shard(vectors, len(chunk), dim, path.name)
        atomic.save_npy(path, vectors)
        built += 1
        per_shard = (time.time() - began) / built
        logger.info(f"{out_path.name}: shard {i + 1}/{n_shards} done, ETA {(n_shards - i - 1) * per_shard / 60:.1f} min")
    part = out_path.with_name(out_path.name + ".part")
    merged = np.lib.format.open_memmap(part, mode="w+", dtype=np.float32, shape=(len(texts), dim))
    for i in range(n_shards):
        vectors = np.load(shard_dir / f"shard_{i:05d}.npy")
        _check_shard(vectors, min(shard_texts, len(texts) - i * shard_texts), dim, f"shard {i}")
        merged[i * shard_texts:i * shard_texts + len(vectors)] = vectors
    merged.flush()
    del merged
    part.replace(out_path)
    atomic.write_json(manifest_path, {**want, "code_commit": code_commit()}, indent=1)
    shutil.rmtree(shard_dir)
    return out_path


def build(raw_dir: Path, out_dir: Path, encode=None, dim: int = DIM) -> dict:
    """Embed every product and distinct query; writes .npy files into out_dir."""
    import pandas as pd
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    products = pd.read_parquet(raw / "shopping_queries_dataset_products.parquet",
                               columns=["product_id", "product_title", "product_bullet_point"])
    examples = pd.read_parquet(raw / "shopping_queries_dataset_examples.parquet",
                               columns=["query_id", "query"])
    q_texts, q_ids = distinct_queries(examples)
    counts = check_counts(len(products), examples["query_id"].nunique(), len(q_texts))
    encode = encode or make_encoder()
    atomic.save_npy(out / "esci_query_ids.npy", q_ids)
    atomic.save_npy(out / "esci_query_texts.npy", np.array(q_texts, dtype=str))
    atomic.save_npy(out / "esci_product_ids.npy", products["product_id"].to_numpy().astype(str))
    embed_sharded(q_texts, encode, out / "esci_queries.npy", dim)
    embed_sharded(product_texts(products), encode, out / "esci_products.npy", dim)
    return counts


def fetch_raw(raw_dir: Path) -> list[Path]:
    """Download the two raw parquet files (about 1.2 GB), resumable."""
    from workloads.q1.datasets import common
    return [common.download_file(f"{RAW_BASE}/{name}", Path(raw_dir) / name) for name in RAW_FILES]
