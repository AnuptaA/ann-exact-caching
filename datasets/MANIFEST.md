# Data manifest

Data files are gitignored. Sizes in bytes. TripClick's terms forbid sharing the data or anything derived from it.

| File | Bytes | Shape | Dtype | SHA-256 | Status |
|---|---|---|---|---|---|
| `datasets/aker/tripclick-ip-docs.npy` | 4687448856 | 1523878x769 | float32 | `773f4c206b0e94456846533e204748f0af3664aeff999367f1a2a6dc732ef57f` | on disk |
| `datasets/aker/tripclick-ip-requests-2018-01-01.npy` | 307600128 | 100000x769 | float32 | `e0ec7513e713603015fb5b3ebf3aff109729e1747ff9c9abe6e883910322b271` | on disk |
| `datasets/tripclick/requests-2018-01-01.json` | 4152556 | - | - | `53c0cd5943816d0de047162734c827c2885e7336a82d0ada77d98a0b0a78b05d` | on disk |
| `medrag-ip-docs.npy` | - | 23888832x769 | float32 | - | expected, cluster only |
| `base_pmids.npy` | - | 23888832 | int64 | - | expected, cluster only |
| `medrag_pmids.npy` | - | 23895135 | int64 | - | expected, cluster only |
| `spacev-10m.npy` | - | 10000000x100 | int8 | - | expected, cluster only |
| `sphere-ip-10m.npy` | - | 10000000x769 | float32 | - | expected, cluster only |
| `spacev-sim-100k-0.3.npy` | - | 100000x100 | int8 | - | expected |
| `spacev-sim-100k-0.6.npy` | - | 100000x100 | int8 | - | expected |
| `spacev-sim-100k-0.99.npy` | - | 100000x100 | int8 | - | expected |
| `triviaqa-ip-sim-100k-0.3.npy` | - | 100000x769 | float32 | - | expected |
| `triviaqa-ip-sim-100k-0.6.npy` | - | 100000x769 | float32 | - | expected |
| `triviaqa-ip-sim-100k-0.99.npy` | - | 100000x769 | float32 | - | expected |

Sources: SPACEV base vectors and streams from Aker's Hugging Face release (`sjoon-oh/aker`); SPHERE and TriviaQA inner-product files, MedRAG and TripClick from the shared Drive folder.
