# Amazon ESCI: Queries, Products, and Top-K Results

Dense retrieval dataset derived from the Amazon Shopping Queries Dataset (ESCI benchmark), encoded with a multilingual sentence-transformer.

## Overview

This dataset contains:
- **130,652 query embeddings** from the full ESCI dataset (EN/ES/JP), encoded with `paraphrase-multilingual-MiniLM-L12-v2`
- **1,814,924 product embeddings** (384-dim, float32), randomly distributed across 100 shards
- **Exact top-K=100 nearest neighbors** (inner product, equivalent to cosine for normalized vectors)
- **Per-product metadata attributes** (locale, brand, color, text lengths, ESCI relevance labels)

All embeddings are 384-dimensional float32 vectors, L2-normalized (unit norm). The similarity metric is **inner product**, which equals cosine similarity for normalized vectors.

## File Format

All files are NumPy `.npz` archives. Load with:

```python
import numpy as np
data = np.load("queries.npz", allow_pickle=True)
```

## Files

### `queries.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `query_vectors` | `(130652, 384)` | float32 | Sentence-transformer query embeddings (L2-normalized) |
| `query_texts` | `(130652,)` | object (str) | Raw query strings |
| `query_ids` | `(130652,)` | int64 | Original ESCI query IDs |
| `num_queries` | scalar | int64 | 130,652 |
| `dim` | scalar | int64 | 384 |

### `passages_100/`

~1.8M product embeddings randomly distributed across 100 shards.

**`manifest.npz`**:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `num_shards` | scalar | int64 | 100 |
| `total_passages` | scalar | int64 | 1,814,924 |
| `dim` | scalar | int64 | 384 |
| `shard_sizes` | `(100,)` | int64 | Number of products per shard (~18K each) |

**`shard_000.npz` through `shard_099.npz`**:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `passage_ids` | `(N,)` | int64 | Global product indices (0-indexed into the product catalog) |
| `embeddings` | `(N, 384)` | float32 | Product embeddings (L2-normalized) |

Each product appears in exactly one shard. Shard assignment is uniformly random (seed=42).

### `topk_100shards.npz`

Exact top-100 nearest neighbor results for all 130,652 queries, computed via brute-force inner product search (FAISS `IndexFlatIP`).

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `topk_ids` | `(130652, 100)` | int64 | Product indices of the 100 nearest neighbors per query |
| `topk_scores` | `(130652, 100)` | float32 | Inner product scores, sorted descending per query |
| `k` | scalar | int64 | 100 |
| `shard_ids` | `(100,)` | int64 | All 100 shard IDs |
| `n_passages` | scalar | int64 | 1,814,924 |
| `distance_metric` | scalar | str | "inner_product" |

Score range: [0.326, 1.000]. Mean top-1 score: 0.793. Mean top-100 score: 0.651.

### `passage_attributes.npz`

Per-product metadata attributes for filtered ANN benchmarking.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `passage_ids` | `(1814924,)` | int64 | Product indices |
| `product_ids` | `(1814924,)` | object (str) | Original ESCI product IDs (e.g., "B07RFN9DL4") |
| `locale` | `(1814924,)` | object (str) | Product locale: "us", "es", or "jp" |
| `locale_id` | `(1814924,)` | int8 | Numeric locale (0=us, 1=es, 2=jp) |
| `brand_id` | `(1814924,)` | int32 | Numeric brand ID (305,160 unique brands) |
| `has_brand` | `(1814924,)` | bool | Whether brand is known (92.1%) |
| `has_color` | `(1814924,)` | bool | Whether color is specified (61.9%) |
| `has_description` | `(1814924,)` | bool | Whether product has a description (51.6%) |
| `has_bullet_point` | `(1814924,)` | bool | Whether product has bullet points (83.2%) |
| `title_length` | `(1814924,)` | int32 | Product title character count (1–400, mean 99) |
| `title_len_bucket` | `(1814924,)` | int8 | Log-scale title length bucket |
| `desc_length` | `(1814924,)` | int32 | Description character count |
| `bullet_length` | `(1814924,)` | int32 | Bullet point character count |
| `n_exact` | `(1814924,)` | int32 | Number of queries where product is an Exact match |
| `n_substitute` | `(1814924,)` | int32 | Number of queries where product is a Substitute |
| `n_complement` | `(1814924,)` | int32 | Number of queries where product is a Complement |
| `n_irrelevant` | `(1814924,)` | int32 | Number of queries where product is Irrelevant |
| `n_judgements` | `(1814924,)` | int32 | Total number of query-product judgements (0–120, mean 1.4) |
| `relevance_ratio` | `(1814924,)` | float32 | Fraction of judgements that are Exact or Substitute |
| `popularity` | `(1814924,)` | int32 | Number of queries this product appears in (proxy for popularity) |
| `pop_bucket` | `(1814924,)` | int8 | Log-scale popularity bucket |
| `num_brands` | scalar | int64 | 305,160 |

Attribute coverage:
- `locale`: 100% coverage. US: 1,215,854 (67.0%), JP: 339,059 (18.7%), ES: 260,011 (14.3%)
- `has_brand`: 92.1% of products have a known brand
- `has_color`: 61.9% of products have a color specified
- `has_description`: 51.6% of products have a description
- `has_bullet_point`: 83.2% of products have bullet points
- ESCI labels: 1,708,158 Exact, 574,313 Substitute, 75,652 Complement, 263,165 Irrelevant

### `raw/`

Original ESCI parquet files from the GitHub repository.

| File | Size | Description |
|------|------|-------------|
| `shopping_queries_dataset_products.parquet` | 1.1 GB | Product catalog (1.8M products) |
| `shopping_queries_dataset_examples.parquet` | 49 MB | Query-product judgements (2.6M rows) |
| `shopping_queries_dataset_sources.csv` | 1.7 MB | Query source annotations |

### `product_embeddings.npy`

Full product embedding matrix (intermediate file, used during pipeline construction).

| Shape | Dtype | Size | Description |
|-------|-------|------|-------------|
| `(1814924, 384)` | float32 | 2.6 GB | All product embeddings in catalog order |

## Usage Example

```python
import numpy as np

# Load queries and results
queries = np.load("queries.npz", allow_pickle=True)
topk = np.load("topk_100shards.npz")

# Get top-10 products for query 0
query_text = str(queries["query_texts"][0])
top10_ids = topk["topk_ids"][0, :10]
top10_scores = topk["topk_scores"][0, :10]

print(f"Query: {query_text}")
for rank, (pid, score) in enumerate(zip(top10_ids, top10_scores)):
    print(f"  #{rank+1}: product {pid}, score={score:.4f}")

# Load a shard
shard = np.load("passages_100/shard_000.npz")
print(f"Shard 0: {shard['embeddings'].shape[0]} products, {shard['embeddings'].shape[1]}-dim")

# Load attributes
attrs = np.load("passage_attributes.npz", allow_pickle=True)
print(f"Locales: {dict(zip(*np.unique(attrs['locale'], return_counts=True)))}")
print(f"Brands: {int(attrs['num_brands']):,} unique")
```

## Key Details

- **Similarity metric**: Inner product (dot product). Since all vectors are L2-normalized, this equals cosine similarity. Higher scores = more similar.
- **Exact search**: Top-K results are computed via brute-force exact search (FAISS `IndexFlatIP`), not approximate nearest neighbors.
- **Embedding model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim). Both queries and products are encoded with the same model.
- **Product text**: Each product is encoded as "title. first_bullet_point" (title + first line of bullet points, separated by a period).
- **Multilingual**: The dataset includes products and queries in English (US), Spanish (ES), and Japanese (JP). The multilingual model handles all three languages in a shared embedding space.
- **Random sharding**: Products are assigned to shards uniformly at random (seed=42).

## Source

- Dataset: [amazon-science/esci-data](https://github.com/amazon-science/esci-data) (Apache-2.0)
- Paper: Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search" (arXiv:2206.06588)
- Embedding model: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
