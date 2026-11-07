# Q1 workloads

Data side of research question 1: K = 10, Euclidean, float32 for indexes and ground
truth. Run from the repo root with `PYTHONPATH=.`; `python -m workloads.q1 --help`.

- `datasets/`: SIFT-10M (HTTP range slice of SIFT-1B), SPACEV 10M and SPHERE 10M (Aker's
  released files, used as is), ESCI (all products, distinct queries, 384-d MiniLM), TriviaQA
  (public questions, de-duplicated, encoded with Sphere's DPR query encoder, not normalized;
  `triviaqa_evidence.py` writes the encoder evidence log, informational only).
- `universe.py`: 10K groups of 50 members, `(1 - lambda) * base + lambda * partner`.
- `streams.py`: Zipf (0.30, 0.60, 0.99), uniform, and no-locality streams; 30/70 history/test split.
- `stats.py`: per-stream statistics (distinct, exact-repeat share, top group, empirical vs
  theoretical Zipf).
- `groundtruth.py`: exact top-100, GPU or numpy candidates, CPU float32 recompute, resumable.
- `distance.py`, `recall.py`: the single distance function and recall@10 with ties (always /10).

Checkpoints and `.pkl` files from third parties are pickles: handle them only in an isolated
environment (see `harness/fidelity.py` for a restricted unpickler).
