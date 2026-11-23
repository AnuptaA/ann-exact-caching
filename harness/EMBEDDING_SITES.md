# Embedding sites

Principle: nothing is re-embedded anywhere in this project except the SPHERE queries; released vectors are used as
released. This file inventories every place where text is embedded or encoded and states what was checked and how.
`tests/harness/test_embedding_sites.py` fails when a new encoder call site appears, so the inventory cannot go stale unnoticed.

How to read the evidence: **code** = read in this repository, **file** = measured on a file on this machine,
**network** = read from a public URL on 2026-11-10, **inferred** = not checked directly.

## 1. Inventory

| # | Place | Input text and source | Model and settings | Output | Released vectors exist? |
|---|---|---|---|---|---|
| A | `workloads/q1/datasets/esci.py` (`make_encoder`, `embed_sharded`, `build`), called by `harness/datasets.py:_esci_embed` in stage `embed` | Products: `product_title + ". " + first line of product_bullet_point` (recipe `product_text`), catalog order, all locales, from `shopping_queries_dataset_products.parquet`. Queries: distinct query texts (first occurrence by `query_id`, then by text) from `shopping_queries_dataset_examples.parquet`, all locales | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384-d, `normalize_embeddings=True`, float32, batch 256, device auto (GPU or CPU), model revision not pinned | `esci_products.npy` (1,814,924 x 384), `esci_queries.npy` (130,193 x 384), each with a `.manifest.json` (recipe, texts hash, commit, device, versions) | No, see section 3 |
| B | `workloads/q1/datasets/triviaqa.py` (`dpr_query_encoder`, `encode_questions`), CLI `triviaqa-encode`, launched by `harness/datasets.py:_sphere_embed` in stage `embed` in the `.venv-encode` interpreter | Public TriviaQA questions (train, validation, test of `rc.nocontext`), stripped and de-duplicated by text (96,005 per `DECISIONS.md`) | Sphere's `dpr_web_query_encoder_hf` (BERT-base CLS, no projection), `max_length=256`, not normalized, float32 | `raw/triviaqa_enc.npy` | Aker released TriviaQA query vectors (`triviaqa-sim-100k-*.npy`, bge-base-en-v1.5 raw CLS). This site is the one exception the principle allows; the choice is investigated in `investigations/sphere_queries/REPORT.md` |
| C | `workloads/q1/datasets/triviaqa_evidence.py` (CLI `triviaqa-evidence`) | 300 TriviaQA questions and 50K Hermes SPHERE_100K passages | dpr_web, and with `--compare` bge-base-en-v1.5 and DPR single-nq | `triviaqa_encoder_evidence.md` (a log) | n/a. Feeds no workload |
| D | `datasets/esci/build_embedding_subsample.py` (older, used by `populate esci` and the simulations) | Same product recipe on a seed-42 random 50K sample; all 130,652 query ids (no text dedup) | Same model, batch 256, normalized, device hard-coded to CPU, no shards, no manifest | `datasets/esci/data_vectors/esci_db50000_s42.npy`, `esci_queries.npy` | Same as A |
| E | Local files `datasets/esci/product_embeddings.npy`, `queries.npz` | Not produced by any script in the repository; `stale/datasets/esci/README_esci.md` documents the model and recipe | Documented as the same model, unit norm | Files only | Same as A |
| F | SPACEV, SIFT-10M, SPHERE base vectors; SPACEV and SIFT-10M query vectors | No text | None | Released files | Yes, used as released (section 2) |
| G | `datasets/tripclick/build_workload.py` (`encode`, steps documents and queries) | TripClick: [title, abstract] pairs of 1.52M MEDLINE documents (documents), normalized search texts of a window of 100,000 consecutive searches from 2018-01-01 (queries) | `ncbi/MedCPT-Article-Encoder` (max 512 tokens, CLS, pairs, bf16 autocast on CPU, cosine to fp32 at least 0.99996 per the script) and `ncbi/MedCPT-Query-Encoder` (max 64 tokens, CLS); fp32 output; inner product turned Euclidean by appending sqrt(M^2 - |x|^2) to documents and 0 to queries (769-d) | `datasets/aker/tripclick-ip-docs.npy`, `tripclick-ip-requests-<start>.npy` | No public release; **not run here, vectors are obtained precomputed** (and, locally, already present under `datasets/tripclick/`). MedCPT is not a TripClick-trained encoder (flagged in `PROGRESS.md`) |
| H | `datasets/reembed_triviaqa_queries.py` (`encode`) | Lines of `triviaqa.txt` (the questions behind Aker's released TriviaQA traces) | `BAAI/bge-base-en-v1.5` raw CLS (only to identify which question each released vector is) and Sphere's DPR_web question encoder (`dpr_web_question_encoder`, BERT CLS, `max_length=256`, uncased, not normalized), float32 | `datasets/aker/triviaqa-dpr-sim-100k-<z>.npy` (same row order as Aker's traces) | Aker's own query vectors exist (bge-base, other space than the SPHERE base). **Not run here, vectors are obtained precomputed.** Same truncation as site B (`max_length=256`) |

No other encoder call exists outside `stale/`, `tests/`, `third_party/` and `investigations/` (code, enforced by the test above).

## 2. Released vectors are used as released (code, tested)

- SPACEV: `spacev-10m.npy` (int8, HF `sjoon-oh/aker`) and the big-ann `query.i8bin` (29,316 queries). SIFT-10M: first 10M rows of `base.1B.u8bin` by byte range and `query.public.10K.u8bin`. SPHERE base: `sphere-10m.npy` (float64, 768-d, not normalized).
- `stage_to_f32` only casts (int8, uint8, float64 to float32) and writes `.f32bin`; no normalization. `test_embedding_sites.py` asserts the cast is exact for all three source types, that the three dataset modules contain no encoder code, and that only `esci` and `sphere` have an `embed` stage (for SPHERE it writes only `triviaqa_enc.npy`, the query vectors).

## 3. ESCI

### Official release (network)
`github.com/amazon-science/esci-data`, directory `shopping_queries_dataset`: `shopping_queries_dataset_examples.parquet` (Git LFS pointer; columns `example_id, query, query_id, product_id, product_locale, esci_label, small_version, large_version, split`), `shopping_queries_dataset_products.parquet` (`product_id, product_title, product_description, product_bullet_point, product_brand, product_color, product_locale`) and `shopping_queries_dataset_sources.csv` (`query_id, source`). The repository also holds training code (`ranking/`, `classification_identification/`, including `compute_bert_representations.py`, which computes BERT representations of query and title at training time). **The release contains no vectors, for queries or for products.** The README states 130,652 unique queries in the large version (2,621,738 judgements) and 48,300 in the reduced version.

### Third-party vector sets (network, not adopted)
A search of the Hugging Face API (datasets matching `esci`, `shopping_queries`, `amazon-esci`, `esci embeddings`; models matching `esci`) found text copies of ESCI, cross-encoder models, and two small vector uploads: `denizberkin/esci-embeddings` (`feature_embeddings.npy`, `query_embeddings.npy`, empty README, MIT, 7 downloads, no stated model, recipe or row order) and `Zeyneperg/esci_full_embeddings` (`all_concat_emb.pt`, no card, 2 downloads), plus tiny `clm-project/*-embeddings` sets (fewer than 1K rows, different data). None is official or widely used (downloads in single digits) and none states a model, so none can be shown to be in the same space as anything else. They are listed only; adopting one needs your decision.

### What the pipeline does (code)
- Query texts: `distinct_queries` takes the examples parquet (all locales, no `small_version`, `large_version` or split filter), `drop_duplicates("query_id")`, sorts by id, then `drop_duplicates("query")`. The harness therefore embeds 130,193 texts for 130,652 query ids (the 130,193 figure is also what the older `queries.npz` yields (file)). `check_counts` compares products and ids with a 3% tolerance, not exactly; the distinct text count is only logged.
- Products: 1,814,924 expected, parquet order, same tolerance. Text recipe above; description, brand and color are not used.
- Vectors are created, not downloaded: `fetch_raw` downloads the two parquet files only (checksums from the LFS pointers). Both sides use the same model by construction.
- Manifest (`embed_sharded`): `recipe` (model, text recipe, normalize, dim), `n`, `texts_sha256`, `code_commit`, and now `made_with` (device, threads, sentence-transformers and torch versions, model revision as reported by the loaded model, max sequence length, batch size, shard size). Shard size and device are not part of the compared recipe, so a CPU-made and a GPU-made file compare equal; shards of different sizes are never mixed. Existing files are reused only after a manifest match or, without a manifest, a 64-row re-encode with cosine >= 0.999.
- The model revision is **not pinned** (`SentenceTransformer(MODEL)` loads the current hub revision); it is now recorded in the manifest, not enforced.

### The older local ESCI files (file)
| File | Shape, dtype | Notes |
|---|---|---|
| `datasets/esci/product_embeddings.npy` | (1,814,924, 384) float32, 2.79 GB | Unit norm (first 2,000 rows 0.9999999 to 1.0000001). Not produced by any script in the repository |
| `datasets/esci/queries.npz` | `query_vectors` (130,652, 384), `query_texts`, `query_ids` 0..130,651 | 130,193 distinct texts; unit norm |
| `datasets/esci/data_vectors/esci_queries.npy` | (130,652, 384) float32 | Bit-identical to `queries.npz` (max abs difference 0.0) |
| `datasets/esci/base_vectors_50k.npy`, symlinked as `data_vectors/esci_db50000_s42.npy` | (50,000, 384) | Equals `product_embeddings[np.sort(rng(0).choice(...))]` (max abs difference 0.0): a seed-0 sample, although the name and `build_embedding_subsample.py` say seed 42 |

The harness never loads these files. Whether `product_embeddings.npy` matches the harness (model, recipe, row order) **cannot be shown**: the raw parquet is not on this machine and no generating script exists. The documented model and recipe are the same, which makes a match plausible, not proven. The proof is a sampled re-encode against the parquet (what `embed_sharded` does for a file without manifest); I did not fetch the 1.2 GB parquet for it. Ask if you want it.

## 4. Where interpolation gets its query vectors (code)
`stage_inputs` and `RequestVectors` read `official.f32bin`, the float32 cast of `spec.official(raw)`. `universe.py` builds members as `(1 - lambda) * base + lambda * partner` with `lambda ~ U(0, 0.5)` from rows of that file, in float32, **not re-normalized**.

| Dataset | Query vectors that interpolation uses | Released? In the base space? |
|---|---|---|
| SPACEV | `query.i8bin` of the big-ann release | Released; same release as the base (inferred from the release, not checked by code) |
| SIFT-10M | `query.public.10K.u8bin` of the SIFT-1B release | Released; same release as the base |
| SPHERE | `triviaqa_enc.npy`: our DPR query encoding of public TriviaQA questions | Not released in this form; the space match to `sphere-10m.npy` is unverified (`DECISIONS.md`, "Not verified yet"). See `investigations/sphere_queries/REPORT.md` |
| ESCI | `esci_queries.npy`: our embeddings of 130,193 distinct query texts | Not released (section 3); same model and recipe family as the products, so the same space by construction |
| tiny | Synthetic (verification only) | n/a |

### What the ESCI finding implies for interpolation
The official release has no vectors for either side, so none of the released-vector cases applies today. For completeness, each case, with what the generator would use, what would have to be created, and the effect on normalization and the metric:

1. **Released vectors for queries and products (not the case today).** The generator would use the released query vectors as `official` and the released product vectors as the base; nothing is created. `fetch` would download vectors (with checksums) and the `embed` stage would go. The query set would be whatever the release defines, not our 130,193 de-duplicated texts. If the released vectors are not unit norm, interpolated members are not unit norm either, and with non-unit product norms the Euclidean ranking of products depends on the query norm: `|q - p|^2 = |q|^2 - 2 q.p + |p|^2`, so scaling `q` changes the weight of `|p|^2` against `q.p`. The "ranking does not depend on the query norm" argument in `DECISIONS.md` holds only for equal-norm products (ESCI today). SPHERE already violates it (unnormalized base).
2. **Vectors for only one side.** (a) Only products released: queries still have to be encoded, with the very model and text recipe that made the products. Without that identity the query space is not the product space; it is checkable only by re-encoding a sample of product texts and comparing cosines (as in Phase 1), and any gap is the SPHERE problem again. The alternative, drawing the "official queries" from product vectors, changes the workload meaning and needs your decision. (b) Only queries released: products must be encoded to match, with the same check.
3. **No released vectors (today).** Both sides must be created, as now (A). The alternative is a third-party set (section 3), which needs the model and row order to be established first; I have not adopted any.

Costs: creating both sides is 1,945,117 texts (1,814,924 products + 130,193 queries); on this machine the CPU estimate is in section 5. Pinning the model revision and keeping the manifest closes the reproducibility gap; adopting a released or third-party set would remove embedding time but only if a same-space product and query set exists, which the evidence does not show.

## 5. Measured CPU rates (this machine, small samples)

Machine: macOS arm64, 8 cores (4 performance, 4 efficiency), 8 GiB RAM, numpy 2.3 with Accelerate, torch 2.15 (scratch
venv), 2026-11-10, a quiet machine (load average about 2). Scripts: `measure/gt_rates.py` and `measure/enc_rates.py`
(kept in the session scratch directory, not in the repository). Samples: ground truth 512 queries against 100,000
base vectors, k = 100, tiles chosen by `fit_cpu_chunks`; encoders 384 synthetic texts (ESCI products about 60 tokens,
TriviaQA-style questions about 12 words).

Ground truth, GFLOP/s of the whole candidate and exact-recompute path (2 x queries x base x dim / seconds):

| dim | 1 thread | 2 | 4 | 8 threads | threads inside BLAS only (8) |
|---|---|---|---|---|---|
| 100 | 8.6 | 14.7 | 20.4 | 25.7 | 8.1 |
| 128 | 9.1 | 8.3 | 25.9 | 29.1 | 11.8 |
| 384 | 38.2 | 59.6 | 85.3 | 111.7 | 44.0 |
| 768 | 79.6 | 117.8 | 128.3 | 171.0 | 78.7 |

The thread pool over query slices (BLAS capped at one thread per worker) is what scales; letting BLAS use all threads
alone does not (the top-k merge and the recompute stay single-threaded). The pool is the default. Small dimensions are
limited by the merge, not the matmul. Accelerate shares matrix units between cores, so the scaling above (2.1x to 3.2x
at 8 threads) is probably worse than on an x86 server with OpenBLAS; `cost.py` uses these numbers, so its estimates
should be conservative there.

Encoders, texts per second:

| encoder | 1 thread | 4 | 8 threads |
|---|---|---|---|
| ESCI MiniLM-L12 multilingual (about 60 tokens) | 70.5 | 124.3 | 137.9 |
| DPR question encoder (about 12 words, max length 64) | 80.7 | 110.6 | 116.8 |

The SPHERE investigation (`investigations/sphere_queries/REPORT.md`) measured 112 questions/s for both bge-base and the DPR_web question encoder on a loaded machine
(`investigations/sphere_queries/REPORT.md`), consistent with the second row.

Extrapolation to full size on this machine at 8 threads (order of magnitude; assumes the rates above hold at full
size, which they do not exactly: real ESCI texts are longer than the synthetic ones, real ground truth reads the base
from disk once per query chunk, and a 10M x 768 float32 base (30.7 GB) does not fit in this machine's RAM):

| dataset | texts to embed | ground-truth queries x base x dim | embedding | ground truth |
|---|---|---|---|---|
| ESCI | 1,945,117 | 640,193 x 1,814,924 x 384 | about 4.5 h | about 2.4 h |
| SPHERE | 96,005 questions | 606,005 x 10M x 768 | about 12 min | about 11.8 h |
| SPACEV | none (released) | 539,316 x 10M x 100 | none | about 12.7 h |
| SIFT-10M | none (released) | 520,000 x 10M x 128 | none | about 14.8 h |

Ground-truth query counts are `10,000 x 50 + 10,000 + official queries`. A GPU would be much faster (not measured here). `make q1-preflight` prints these estimates for the machine it runs on, and each
stage prints its own at the start.
