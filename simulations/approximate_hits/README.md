# Calibrated approximate caching: the headline and timing experiments

This folder holds the experiments behind the paper's two main plots:

1. **Headline plot**: the cache hit rate each technique reaches while keeping the promised
   retention, with an exact backend.
2. **Timing plot**: end-to-end mean latency when each technique sits in front of the same tuned
   DiskANN index.

The full decision log, with every number, failure and fix, is
`docs/approximate_cache_hits_context.md`. Section numbers like "9u" below refer to it.

## What is being measured

- **Retention.** A cache hit returns the cache's answer instead of asking the backend. A hit's
  retention is the share of the backend's top-10 that the served answer contains (doc 9o). The
  promise is that the mean retention over hits is at least T = 0.9. With an exact backend,
  retention is recall.
- **Ours.** A greedy cache built in warm-up from the first 20,000 requests. A logistic score per
  request (7 features: how well the best-fitting cached balls cover the query, and how many
  uncached "shell" vectors intrude). The hit cutoff on that score is chosen by a fixed-order
  sequence of one-sided normal tests on calibration labels, which gives the guarantee
  (`threshold_selection.py`). The walk starts at the most certifiable threshold of an independent
  fit split (doc 9p). No knob to tune per workload.
- **Baselines.** All are given the same memory budget (cache size as a share of the database:
  0.3%, 1%, 3%, 10%).
  - LRU: our hit rule, with every miss becoming an entry and least-recently-used eviction.
  - Aker: the authors' C++ code (Standard and Proximity builds) through a C shim.
  - QVCache: a reimplementation following its paper.

  Aker and QVCache are each shown twice: tuned by hand to reach 0.9 (best hit rate among a
  settings grid, chosen after the fact), and with their paper's own setting (doc 9q).
- **Workloads.** Aker's simZipf traces on SPACEV-10M (Zipf 0.3 / 0.6 / 0.99), the same on
  TriviaQA over SPHERE-10M (queries re-embedded with the corpus' DPR encoder, inner product made
  Euclidean; doc 9b, 9c, 9l), the SPACEV real query log, and ESCI (1M products, no repeated
  queries).

## Setup

Two virtual environments at the repository root:

| env | contains | runs |
|---|---|---|
| `.venv` | numpy, torch, transformers, scikit-learn | `online_sim.py`, preps, plots |
| `.venv-ann` | faiss-cpu 1.8, diskannpy 0.7, numpy 1.25 | real Aker, DiskANN, timing |

Real Aker: build Boost and FAISS into `~/third_party/{boost-install,faiss-install}`, clone Aker
to `~/third_party/aker` (commit 6bb326a) and a second copy to `~/third_party/aker_prox`, then run
`sh aker_real/build.sh`. That builds `libakershim.so` (Standard mode) and `libakershim_prox.so`
(Proximity mode). Jobs set `LD_LIBRARY_PATH` to the right `build/lib`.

Run everything from the repository root with `PYTHONPATH=.` and `OMP_NUM_THREADS=1`.

## Data (once)

```sh
cd datasets/aker && sh download.sh && cd -        # SPACEV-10M, SPHERE-10M, Aker traces, SPACEV1B logs
.venv/bin/python datasets/reembed_triviaqa_queries.py   # TriviaQA queries -> DPR_web space (doc 9b)
.venv/bin/python datasets/make_sphere_ip_transform.py   # 769-d inner-product transform (doc 9c)
.venv/bin/python datasets/esci/build_embedding_subsample.py   # ESCI 1M (see datasets/esci/README_esci.md)
```

Per workload, a prep folder with exact top-500 / top-1000 neighbors and request streams:

```sh
.venv/bin/python simulations/approximate_hits/online_prepare.py --dataset trace \
    --trace spacev-aker-0.6 --out simulations/approximate_hits/results/online/spacev-aker-0.6
.venv/bin/python simulations/approximate_hits/online_prepare.py      # ESCI (default) -> results/online/esci1m
```

Trace names: `spacev-aker-{0.3,0.6,0.99}`, `triviaqa-ip-{0.3,0.6,0.99}`, `spacev-real`. The
registry is `WORKLOADS` in `run_e2e_oracle_trace.py`.

## Headline plot

The job generators print one shell command per line. Each command skips itself if its result
JSON exists.

```sh
S=simulations/approximate_hits
# ours (greedy_score_rebuild), LRU (lru_k10_none, lru_k20_none, lru_k20_all), QVCache grid;
# seeds 0 and 1, per cache size (--budget; 1% is the default)
.venv/bin/python $S/online_round2_jobs.py --budget 0.1 --preps spacev-aker-0.6 ...
# real Aker: the original grid (--only stable) and the settings tuned to reach 0.9
#   (--tuned: Standard alpha_T .1 / .01 x alpha_L 1 / 1.1; Proximity --prox-frac .001-.3)
.venv/bin/python $S/aker_real/aker_real_jobs.py --preps spacev-aker-0.6 --only stable --budgets 0.003 0.01 0.03 0.1
.venv/bin/python $S/aker_real/aker_real_jobs.py --preps spacev-aker-0.6 --tuned --budgets 0.003 0.01 0.03 0.1
# then
.venv/bin/python $S/plot_headline.py --out $S/results/summary/headline_draft_v4.png
```

- Only the configurations the plot reads need to run. The launchers filter the generator's output
  with `KEEP` (see `run_headline_rerun_fitstart.sh`, `run_headline_extra_sizes.sh`,
  `run_headline_esci.sh`, `run_headline_complete.sh`).
- ESCI uses `--stream random`, and real Aker on ESCI gets `--stream random` substituted.
- Outputs:
  - `results/online/<prep>/r2_<stream>[_b<budget>]_s<seed>/<config>.json` (simulator);
  - `results/aker_real/<prep>/trace_b<budget>/<setting>.json` (real Aker).
- Plot rule per point: for ours and LRU, the mean of two seeds. For Aker and QVCache, the best hit
  rate among settings with retention >= 0.89. Hollow, faded points mean no setting reached it.

## Timing plot

Every technique sits in front of the same DiskANN index per workload. Misses go to DiskANN, and
cache updates count as background work (doc 9l). The cache is driven through the whole stream
after warm-up using DiskANN's stored answers, which equal live searches exactly. Every 20th
request is timed end to end with live DiskANN calls (doc 9m). The remaining arms are DiskANN
alone, and DiskANN with the cache's memory as its node cache.

The whole chain is `run_backend_redo.sh` (doc 9u). Each step waits on the previous step's
outputs:

1. **Indexes.** `eval_real_backend.py --index diskann --build-only` (SPACEV `spacev10m`, ESCI
   `esci1m`, TriviaQA `sphereip10m-pq256`), plus the variants in `index_study.py`: the Aker
   paper's settings `spacev_pq110` and `aker_tqa`, and `pq768`, `r128`, `mem`.
2. **Quiet probe.** `latency_probe.py` measures, per index, recall@10 on 5,000 queries and
   single-query latency for search list x beam width 2-32. Run through `run_timing_batch2.sh
   probe ...` so the machine is quiet. On this disk the beam width is the dominant lever (doc 9u).
3. **Selection.** `select_backend.py` picks, per workload, the index / search list / beam width
   with the lowest latency at recall 0.95 (0.9 if 0.95 needs more than 100 ms). Writes
   `results/backends/<prep>/choice.json`.
4. **Stored answers** at exactly that setting: `eval_real_backend.py --fixed <target>:<L> --beam
   <W> --index-dir <dir>`. Writes `results/backends/<prep>/diskann.{json,npz}`. The harness takes
   the beam, index and target from them.
5. **Warm-up states** for ours and LRU: `online_sim.py --method greedy --save-state ...
   --backend-file results/backends/<prep>/diskann.npz --shell-scope balls --shell-balls 10`.
   They go in `results/timing_v4/<prep>/state_b<b>.npz`.
6. **Timing runs.** `timing_jobs.py --set final` produces the job lines. `eval_timing_online.py`
   does one run: `--method ours|lru|aker|qvcache`, `--sample-every 20`, `--maintenance
   background`. They run through `run_timing_batch2.sh final ...`, which holds the queue,
   suspends every other job and runs one timing job at a time. Each run checks for a quiet
   machine (`timing_env.py`: other processes <= 1 core and <= 50 MB/s) and records a verdict.
7. **Plot.** `.venv/bin/python $S/plot_timing.py` writes `results/summary/timing_draft_v3.png`
   and `timing_table.md`.

Fidelity notes:
- In the harness, ours is static after warm-up (stationary workloads: no recalibration, no
  audits). LRU recalibrates, with 10% of hits audited.
- The in-cache search is FAISS HNSW with search list 256, with an exact fallback if it returns too
  few neighbors (doc 9n, 9o).
- The harness's LRU agrees with the simulator's within about a point of hit rate.

## Running many jobs: the queue

Long jobs run detached under a small memory-aware queue (doc 9i, 9q, 9t):

- `mx_run_job.sh <ignored> '<line>'` runs one job line:
  - admission from the machine's actual free memory plus the expected growth of running jobs;
  - memory estimates learned per job class (`/tmp/mx/peaks.tsv`);
  - jobs killed for memory are retried, with back-off;
  - priority `MX_PRIORITY=plot|adhoc`, with ad-hoc work killed first.
- `mx_complete.sh <list> <flag> [slots]` runs a list until every output exists, then sets the flag.
- `mx_watchdog4.sh <flags>...` handles the emergency kill order (ad hoc first), adapts the
  growth-reservation factor, and supervises launchers.
- Status: `/tmp/mx/status.log`, `/tmp/mx/failures.log`, `/tmp/mx/watchdog.log`. `/tmp/mx/HOLD`
  pauses admissions.

Example: `setsid nohup sh $S/mx_complete.sh jobs.txt /tmp/mx/MY.DONE 24 > /tmp/mx/my.log 2>&1 &`

## Other files

- `online_sim.py` also supports experiments outside the two plots: data changes, drift
  detection and shadow rebuilds, simulated inexact backends, probes.
- `online_baselines.py` (Aker and QVCache reimplementations), `greedy_population.py`,
  `eval_core.py`, `eval_generalize.py` and the `run_e2e_oracle_*.py` loaders are used by the
  simulator.
- The other `eval_*`, `summarize_*` and `plot_*` scripts are the earlier offline analysis rounds
  (docs, sections 1-8). They are not needed for the two plots.

## Tests

```sh
.venv/bin/python -m pytest tests/approximate_hits/
```
