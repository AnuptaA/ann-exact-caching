"""ESCI: the full product set and distinct query texts, same 384-d model as before.

Products are embedded in catalog order from "title. first_bullet"; embeddings are
L2-normalized float32, as in datasets/esci/README_esci.md. There is no official
query set, so the official queries are the distinct query texts.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_PRODUCTS = 1_814_924
EXPECTED_QUERY_IDS = 130_652
COUNT_TOLERANCE = 0.03
DIM = 384


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


def build(raw_dir: Path, out_dir: Path, encode=None) -> dict:
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
    np.save(out / "esci_queries.npy", encode(q_texts))
    np.save(out / "esci_query_ids.npy", q_ids)
    np.save(out / "esci_query_texts.npy", np.array(q_texts, dtype=str))
    np.save(out / "esci_products.npy", encode(product_texts(products)))
    np.save(out / "esci_product_ids.npy", products["product_id"].to_numpy().astype(str))
    return counts
