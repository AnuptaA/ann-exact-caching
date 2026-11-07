# TriviaQA encoder evidence (informational)

- passages: 50000 Hermes SPHERE_100K passages (sphere100k_0.parquet)
- questions: 300 TriviaQA questions with answer aliases (tqa_val.parquet)
- vectors: raw, unnormalized
- top: 20
- lift: top-20 answer-passage precision divided by the rate for random passages

## Euclidean distance on raw vectors (the Q1 metric)
| encoder | questions | top-20 precision | lift | any hit |
|---|---|---|---|---|
| dpr_web (Sphere query encoder, used) | 300 | 0.043 | 12.3x | 0.287 |
| bge-base-en-v1.5 (Aker's payload encoder, raw CLS) | 300 | 0.041 | 11.7x | 0.273 |
| dpr single-nq (public DPR question encoder) | 300 | 0.013 | 3.7x | 0.130 |

random-passage rate: 0.0035

## Inner product on raw vectors (DPR's training metric)
| encoder | questions | top-20 precision | lift | any hit |
|---|---|---|---|---|
| dpr_web (Sphere query encoder, used) | 300 | 0.069 | 19.7x | 0.393 |
| bge-base-en-v1.5 (Aker's payload encoder, raw CLS) | 300 | 0.045 | 13.0x | 0.337 |
| dpr single-nq (public DPR question encoder) | 300 | 0.047 | 13.4x | 0.343 |

random-passage rate: 0.0035
