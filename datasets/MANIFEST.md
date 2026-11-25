# Data manifest

Data files are gitignored. TripClick's terms forbid sharing the data or anything derived from it. Each Drive
dataset has its own folder with a README; `scripts/fetch_data.sh` takes the files from these folders by name and
stops when one is missing. Set `RECAL_DRIVE_<DATASET>` to the dataset's folder link (shared privately), in the environment or in `.env.local`. Real downloads run on the cluster.

## Sources

| Dataset | Source | Files |
|---|---|---|
| SPACEV | Hugging Face `sjoon-oh/aker` | `spacev-10m.npy` (10,000,000 x 100 int8), `spacev-sim-100k-<z>.npy` (100,000 x 100 int8), Zipf 0.3, 0.6, 0.99 |
| SPHERE base | Hugging Face `sjoon-oh/aker` | `sphere-10m.npy` (10,000,000 x 768 float64, 61.4 GB) |
| TriviaQA | Drive folder (`RECAL_DRIVE_TRIVIAQA`) | `triviaqa-ip-sim-100k-<z>.npy` (request vectors, 769-d float32), `triviaqa-dpr-sim-100k-<z>.npy` (768-d, before the transform), `-lineidx.npy`, `.json` (match statistics), `sphere-ip-10m.json` (M and norm statistics) |
| MedRAG corpus | Drive folder (`RECAL_DRIVE_MEDRAG`) | `medrag-ip-docs.npy`, `base_pmids.npy`, `base.json`, `medrag_pmids.npy` |
| MedRAG requests (100K) | Drive folder (`RECAL_DRIVE_MEDRAG_100K`) | `medrag-ip_100k-requests.npy`, `requests_100k.json`, `zipf_draw_100k.json`, `paraphrases_100k.json`, `paraphrase_stats_100k.json` |
| LAION-prompts requests | Drive folder (`RECAL_DRIVE_LAION_PROMPTS`) | `rc-diffusiondb-requests.npy` (100,000 x 512 float32, stream order), `q.npy` (84,416 x 512 float32), `streams.npz` (key `trace`), `top128.npy` (84,416 x 128 int32, exact), `row_to_passage_id.npy` (9,646,707 int64), `requests.json`, `shuffle_perm.npy`, `build.json`, `checks.json`, `README.md` (kept as `README_prompts.md`) |
| LAION-10M corpus | Drive folder (`RECAL_DRIVE_LAION`, the LAION folder or its `data_vectors`) | `data_vectors/shard_000.npz` ... `shard_099.npz` (keys `passage_ids` int64 (N,), `embeddings` float16 (N, 512); 95,494 to 97,252 rows each, 9,646,707 in total), `data_vectors/manifest.npz` (`shard_sizes`, `total_passages`, `dim`); the folder also holds `queries.npz`, `topk_{1,10,100}shards.npz`, `passage_attributes.npz` and `README_laion.md`, which the pipeline does not use. The build writes `datasets/aker/laion-ip-docs.npy` (float32, same row order) with its SHA-256 in `laion-ip-docs.json` |
| TripClick | Drive folder (`RECAL_DRIVE_TRIPCLICK`) | `tripclick-ip-docs.npy`, `tripclick-ip-requests-2018-01-01.npy`, `requests-2018-01-01.json`; raw 768-d shards and `LICENSE.md` are not fetched |

## Transforms

- Inner product to Euclidean (SPHERE, MedRAG, TripClick): every corpus vector x gets sqrt(M^2 - |x|^2) appended, M the
  largest corpus norm, and every request gets 0, so Euclidean neighbors equal inner-product neighbors (769-d).
- The SPHERE inner-product corpus `sphere-ip-10m.npy` (10,000,000 x 769 float32, 30.8 GB) is not on Drive: it is built
  from `sphere-10m.npy` by `scripts/build_sphere_ip.py` and its M must equal the one in `sphere-ip-10m.json`.
- TriviaQA requests are Aker's TriviaQA traces with each request re-embedded by SPHERE's DPR_web question encoder
  (CLS, at most 256 tokens). Request counts: Zipf 0.3: 99,903; 0.6: 99,990; 0.99: 99,907.
- MedRAG: 23,888,832 articles (MedCPT article embeddings, 768-d before the transform) in `base_pmids.npy` order;
  requests are MedCPT query-encoder embeddings (at most 64 tokens). The 100K stream is 100,000 independent Zipf 0.8
  draws (seed 0) over the 500 PubMedQA questions (counts: maximum 7,602, median 92, minimum 38), every occurrence a
  unique rewording by Claude Haiku 5.5 (Amazon Bedrock). The earlier 10K stream (rewordings by another model) is not
  fetched unless `--medrag-10k` is given and is not used by the experiments.
- TripClick: 1,523,878 MEDLINE documents (MedCPT article encoder, at most 512 tokens) and 100,000 consecutive searches from
  2018-01-01 (MedCPT query encoder, at most 64 tokens), in time order.

## Files

Sizes in bytes; `-` means not yet on this machine.

| File | Bytes | Shape | Dtype | SHA-256 | Status |
|---|---|---|---|---|---|
| `datasets/aker/tripclick-ip-docs.npy` | 4687448856 | 1523878x769 | float32 | `773f4c206b0e94456846533e204748f0af3664aeff999367f1a2a6dc732ef57f` | on disk |
| `datasets/aker/tripclick-ip-requests-2018-01-01.npy` | 307600128 | 100000x769 | float32 | `e0ec7513e713603015fb5b3ebf3aff109729e1747ff9c9abe6e883910322b271` | on disk |
| `datasets/tripclick/requests-2018-01-01.json` | 4152556 | - | - | `53c0cd5943816d0de047162734c827c2885e7336a82d0ada77d98a0b0a78b05d` | on disk |
| `spacev-10m.npy` | - | 10000000x100 | int8 | - | expected, cluster only |
| `sphere-ip-10m.npy` | - | 10000000x769 | float32 | - | expected, cluster only |
| `spacev-sim-100k-0.3.npy` | - | 100000x100 | int8 | - | expected |
| `spacev-sim-100k-0.6.npy` | - | 100000x100 | int8 | - | expected |
| `spacev-sim-100k-0.99.npy` | - | 100000x100 | int8 | - | expected |
| `triviaqa-ip-sim-100k-0.3.npy` | - | 99903x769 | float32 | - | expected |
| `triviaqa-ip-sim-100k-0.6.npy` | - | 99990x769 | float32 | - | expected |
| `triviaqa-ip-sim-100k-0.99.npy` | - | 99907x769 | float32 | - | expected |
| `medrag-ip-docs.npy` | - | 23888832x769 | float32 | - | expected, cluster only |
| `base_pmids.npy` | - | 23888832 | int64 | - | expected, cluster only |
| `medrag_pmids.npy` | - | 23895135 | int64 | - | expected, cluster only |
| `medrag-ip_100k-requests.npy` | - | 100000x769 | float32 | - | expected |

- LAION-prompts: unit-norm CLIP ViT-B/32 vectors, so inner product, cosine and Euclidean rank the same and no coordinate is appended. `scripts/laion_data.py verify` checks shard rows (sum 9,646,707), `row_to_passage_id` against the shards' passage ids, request and query shapes, stream indices, request rows equal to `q.npy[stream]`, and norms. Corpus row r is the r-th vector of the shards in order. Hubness is strong (the 1,000 most frequent images fill 16.8% of top-10 slots) and distances are flat (nearest at about 0.985 of the 10th's distance).
