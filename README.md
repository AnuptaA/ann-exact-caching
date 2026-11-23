# ReCal

ReCal is a recall-certified result cache for top-K vector search. It keeps a mini cache for exact repeats and a main cache of balls around past queries, and answers a query from the cache only when a calibrated score certifies that the expected recall of the answer meets the user's target.

This repository holds ReCal's simulators, the benchmark harness that compares it with other result caches (Aker, QVCache, Proximity) over disk-based backends (DiskANN, pgvector), and the workload builders. Code identifiers, flags and result keys still use the older names (for example `ours`, `greedy_score_rebuild`, `cachecore`); they were not renamed so that stored results and scripts keep working.

Datasets are not in git. TripClick's terms forbid redistributing the data or any derived data: never commit, upload or copy it (`datasets/tripclick/` is gitignored except the scripts).

## Repository map

| Path | What it is |
|---|---|
| `simulations/approximate_hits/` | The simulation pipeline: the earlier cache simulator (`online_sim.py`), the C++ cache core, baselines, backends, timing and throughput, job runner. See `simulations/approximate_hits/README.md` for the headline and timing experiments |
| `simulations/approximate_hits/online_sim.py` | Online simulator and per-request timed mode (methods static, periodic, triggered, lru, ogreedy, aker, qvcache) |
| `simulations/approximate_hits/online_baselines.py` | Python Aker and QVCache simulations (and `QVCacheCore`) |
| `simulations/approximate_hits/cachecore/` | C++ cache core (pybind11, vendored hnswlib 0.8.0): `cachecore.cpp`, `qvcache.h`, `build.sh` |
| `simulations/approximate_hits/core_cache.py` | Adapter between `online_sim.Cache` and the C++ core |
| `simulations/approximate_hits/aker_real/` | Real Aker library built from source, C shim (`aker_shim.cc`), patch, jobs and evaluation |
| `simulations/approximate_hits/eval_real_backend.py`, `select_backend.py`, `latency_probe.py`, `workload_recall.py`, `node_cache_curve.py` | DiskANN backend: build, tuning of search list and beam, recall, node cache |
| `simulations/approximate_hits/eval_timing_online.py`, `timing_jobs.py`, `jobs_v6.py`, `timing_env.py` | Timing runs (live and composed), job generators, quiet-machine guard |
| `simulations/approximate_hits/throughput_{replay,all,backend}.py` | Closed-loop throughput replay against real DiskANN processes |
| `simulations/approximate_hits/run_jobs.py`, `mx_*.sh`, `run_*.sh` | Parallel job runner, memory-aware queue, watchdogs, experiment drivers |
| `simulations/approximate_hits/plot_headline.py`, `plot_timing.py` | Headline (hit rate) and timing plots |
| `simulations/approximate_hits/{eval_*,plot_*,summarize_*}.py` | Earlier offline evaluations and figures |
| `harness/` | Q1 harness: pgvector backend, precomputed answer tables, Aker and Proximity replay (`harness/cpp/ak_replay.cc`), deploy scripts. See `harness/README.md` |
| `workloads/q1/` | Q1 workload generator: SPACEV, SPHERE, ESCI, TriviaQA datasets; Zipf, uniform, no-locality streams; exact ground truth with CPU or GPU |
| `third_party/aker/` | Vendored Aker library with local patches (`PATCHES.md`) used by `harness/` |
| `datasets/` | Builders: `medrag/`, `tripclick/`, `esci/`, `synthetic/`, `reembed_triviaqa_queries.py`, `make_sphere_ip_transform.py` |
| `utils/` | `utils/base/{io,knn,logs}.py` and `utils/cache/` (helpers of the harness) next to the older exact-cache modules (`utils/base/{lemmas,kv_cache,...}.py`) |
| `tests/` | `approximate_hits`, `test_cachecore.py`, `harness`, `workloads`, `base`, `cache`, `utils`, `esci`, `synthetic` |
| `investigations/sphere_queries/` | SPHERE query-encoder investigation (report in `REPORT.md`) |
| `stale/` | Old exact-cache code and docs, kept for reference |
| `envs/`, `env.example` | Requirement files per environment, environment variables |
| `docs/approximate_cache_hits_context.md` | Running decision log of the cache experiments |

## Environments and builds

Requirement files are in `envs/` (see `envs/README.md`). Create them at the repository root:

| Environment | Python | Used for |
|---|---|---|
| `.venv-core` | 3.11 | cache core build (`envs/core.txt`), online simulator on the core, timing |
| `.venv-ann` | any | real Aker shim, DiskANN, throughput (`envs/ann.txt`) |
| `.venv` | any | dataset preparation, embeddings, offline evaluations, plots (`envs/prep.txt`) |
| harness venv | 3.10+ | `harness/`, `workloads/q1/` (`envs/harness.txt`, pinned in `requirements-q1.lock`) |

Builds:
- C++ core: `sh simulations/approximate_hits/cachecore/build.sh` (needs `.venv-core` with pybind11; `CORE_ARCH` overrides `-march=native`). Test: `.venv-core/bin/python -m pytest tests/test_cachecore.py`.
- Real Aker: build Boost and FAISS 1.8.0 into `$RECAL_THIRD_PARTY` (default `~/third_party`), then `sh simulations/approximate_hits/aker_real/build.sh` (builds the vendored `third_party/aker`, a Standard and a Proximity copy and `libakershim*.so`; `AKER_SRC=clone` builds the authors' clone at `6bb326a` instead, whose filter parameters cannot be set).
- Harness: `make q1-setup` on an Ubuntu x86-64 instance (PostgreSQL 16, pgvector 0.8.0, FAISS, Boost, drivers), or `make build-replay` locally. Run book: `harness/README.md`.
- Environment variables: `env.example`.

Run scripts from the repository root with `PYTHONPATH=.` and `OMP_NUM_THREADS=1`. Shell drivers find the root themselves (`RECAL_REPO` overrides).

## Data layout

All paths are relative to the repository root and gitignored.

| Workload | Files | Built by |
|---|---|---|
| SPACEV (Aker traces, Zipf 0.3 / 0.6 / 0.99) | `datasets/aker/spacev-10m.npy`, `datasets/aker/spacev-sim-100k-<z>.npy` | Aker's released files (`datasets/aker/download.sh` is referenced by `simulations/approximate_hits/README.md` but is not in the repository) |
| SPACEV real log | `datasets/aker/spacev1b/{query_log.bin,query.bin}` | Aker's released files |
| TriviaQA over SPHERE | `datasets/aker/sphere-10m.npy`, `sphere-ip-10m.npy`, `triviaqa-ip-sim-100k-<z>.npy` | Aker's files, then `datasets/reembed_triviaqa_queries.py` and `datasets/make_sphere_ip_transform.py` (needs the DPR encoders in `$RECAL_THIRD_PARTY/sphere/`) |
| MedRAG | `datasets/aker/medrag-ip-docs.npy` (about 74 GB), `datasets/aker/medrag-ip-requests.npy`, `datasets/medrag/*` | `datasets/medrag/download_medcpt.sh`, `medrag_pmids.py`, `build_workload.py draw|base|queries`, `paraphrase.py` (local llama.cpp server on 127.0.0.1:18080) |
| TripClick | `datasets/tripclick/raw/` (Hugging Face token needed), `datasets/aker/tripclick-ip-docs.npy`, `tripclick-ip-requests-2018-01-01.npy` | `datasets/tripclick/download.py`, `build_workload.py --start 2018-01-01 --n 100000` |
| ESCI 1M | `datasets/esci/data_vectors/{esci_db1000000_s42.npy,esci_queries.npy}` | `datasets/esci/build_embedding_subsample.py` (see `datasets/esci/README_esci.md`) |
| Q1 streams (SPACEV, SPHERE, ESCI, TriviaQA; Zipf, uniform, no-locality) | `$Q1_DATA/<dataset>/` (`base.f32bin`, `streams_seed0.npz`, `gt/`, `answers/`, ...), default `/data/q1` | `make q1-generate DATASET=<name>` |
| Per-workload preps and backends | `simulations/approximate_hits/results/online/<prep>/` and `results/backends/<prep>/` | `online_prepare.py`, `eval_real_backend.py` |

## Running things

Backends:
- DiskANN: `eval_real_backend.py --index diskann --build-only`, `latency_probe.py`, `select_backend.py`, then `eval_real_backend.py --fixed <target>:<L> --beam <W>`; the chain is `run_backend_redo.sh` (see `simulations/approximate_hits/README.md`, "Timing plot"). Search list and beam are chosen per workload for recall 0.95.
- pgvector: `make q1-pg-start`, `make q1-generate DATASET=<name>` (loads, indexes, tunes `ef_search` for recall 0.95, writes the answer table); `make q1-smoke` runs the stack on a tiny dataset.

Baselines:
- Aker (real library through the shim): `aker_real/aker_real_jobs.py --preps <prep> --only stable|--tuned --budgets ...`, evaluated by `aker_real/eval_aker_real.py`.
- Aker and Proximity (replay over the pgvector answer table): `make q1-run-aker-b{1,5,10}`, `make q1-run-proximity-p0.1-b{1,5,10}`, `make q1-run-proximity-p1-b{1,5,10}`, `make q1-run-all`, `make q1-collect`.
- Proximity mode: Aker built with `AKER_ENABLE_PROXIMITY_MODE` (`libakershim_prox.so`), `--prox-frac` on `eval_aker_real.py`.
- QVCache: `online_sim.py --method qvcache` (Python or core, `--cache-impl core`); timed variant `eval_timing_online.py --method qvcache`.

Workloads and experiments:
- Prep: `online_prepare.py --dataset trace --trace <name> --out results/online/<name>` with `<name>` one of `spacev-aker-{0.3,0.6,0.99}`, `triviaqa-ip-{0.3,0.6,0.99}`, `spacev-real`, `tripclick-ip`, `medrag-ip` (registry: `WORKLOADS` in `run_e2e_oracle_trace.py`); ESCI: `online_prepare.py` (default).
- Headline (hit rate): `online_round2_jobs.py`, `aker_real/aker_real_jobs.py`, `jobs_v6.py --set headline-exact`, `plot_headline.py`. Timing: `jobs_v6.py --set timing`, `timing_jobs.py`, `run_timing_batch2.sh`, `plot_timing.py`. Throughput: `throughput_replay.py` (`--io-complete`, `--nodes`). Details: `simulations/approximate_hits/README.md` and the decision log.
- New workloads (`tripclick-ip`, `medrag-ip`) use `run_headline_new_workload.sh` and `run_timing_new_workload.sh`, which read job-list templates from the queue directory (`$RECAL_QUEUE_DIR`, default `/tmp/mx`); those lists are not in the repository.

Job runner: job generators print one shell command per line and each command skips itself when its result JSON exists. `run_jobs.py <file> --workers N --cores-per-job C --first-core F` runs the lines with `taskset` pinning (rewriting `.venv-core/bin/python`). `mx_run_job.sh` admits a job only when free memory allows it (learned per-class peaks in `$RECAL_QUEUE_DIR/peaks.tsv`, retries after out-of-memory kills, `HOLD` and `skip.txt` files); `mx_complete.sh`, `mx_requeue.sh` and `mx_watchdog4.sh` re-run missing jobs and supervise. Timing runs go through `run_timing_batch2.sh`, which suspends other jobs and checks that the machine is quiet. The runner assumes a large Linux host with `/proc`.

Tests (no data or services): `.venv/bin/python -m pytest tests/approximate_hits tests/base tests/cache tests/utils tests/workloads tests/harness`. `make test-q1` runs the harness and workload tests. The old exact-cache suites `tests/esci` and `tests/synthetic` are older suites; they are excluded from the default collection (`pytest.ini`) and currently fail. Run them explicitly by path.

## ReCal implementation (`recal/`)

`recal/` implements ReCal (C++ core `recal/core/`, Python construction, hit policy, updates and rebuild signals); `recal/README.md` maps each component to its code and test. The earlier cache implementation and drivers in `simulations/approximate_hits/` are kept for comparison (`--method prototype` selects it).

- Build and test: `sh recal/core/build.sh` and `.venv-core/bin/python -m pytest tests/recal` (Python 3.11; `envs/core.txt`).
- Run a cell: `PYTHONPATH=. .venv-core/bin/python -m recal.cells --method recal|prototype|aker|qvcache --prep <prep> --stream <name> --budget 0.05 --out <dir>`; `python -m recal.jobs` prints job lines for `simulations/approximate_hits/run_jobs.py`.
- Budget rule: every system gets the same byte budget M, a percentage of the dataset's stored bytes (N x d x bytes per coordinate), and is charged for everything it keeps in RAM to serve: vectors at their stored precision, ids, graphs and indexes, entry metadata, models and its own caches. Every result JSON carries `bytes` by structure, `M` and `within_M`.
- Baseline settings: Aker at the paper's settings (alpha_T 0.25, alpha_L 1.1, delta 0, FIFO, query-filter HNSW m 4, efConstruction 8, efSearch 16; the filter values need the shim built against the vendored library, which is the default of `sh simulations/approximate_hits/aker_real/build.sh`; a run whose FILTER line differs fails unless `--allow-unverified-filter`; `--legacy-defaults` restores the authors' bootstrap 0.9 and delta 5); Proximity is Aker's fixed-threshold mode with the 0.1 and 1 percentiles of the squared distances between distinct history queries (`--prox-percentile`); QVCache with D 0.075, alpha 0.9 and four mini-indexes, lookup and insertion per request.
- Measurement options (each behind a flag that restores the old behaviour): throughput replay aligned to recorded request ids and failing on dead workers; page cache evicted between arms (`posix_fadvise`) with randomized arm order; every composed-timing miss charged a backend latency; the quiet guard fails closed; result files written atomically and stamped with a config hash so stale results are not reused; core ranges from the machine; cache-core crash paths fixed.
- Parts that need the Linux cluster (DiskANN, the real Aker library and Proximity mode, `mincore` and `fadvise`, `/proc`-based queue and guard, MedRAG, SPACEV and TriviaQA at 10M) are implemented and mock-tested.

## Fetching the precomputed TripClick and MedRAG vectors

The vectors are on a shared Google Drive folder (ask the maintainers for access). Use only the TripClick and MedRAG files and do not re-share them.

```sh
python3 -m venv .venv-fetch && .venv-fetch/bin/pip install gdown
sh scripts/fetch_drive_data.sh --local --dry-run   # list what would happen (add --listing file.json to stay offline)
sh scripts/fetch_drive_data.sh --local             # TripClick and MedRAG's small files: needs about 0.5 GB and 10 GB free
sh scripts/fetch_drive_data.sh --cluster           # adds the MedRAG corpus: about 74 GB more (Linux cluster only)
```

The script downloads by file id (never the whole folder), refuses to start if fewer than 10 GB would remain free, places files where the workload registry expects them (`datasets/aker/`, `datasets/medrag/`, `datasets/tripclick/`) and runs `scripts/verify_data.py`. `datasets/MANIFEST.md` records name, size, shape, dtype and SHA-256.
