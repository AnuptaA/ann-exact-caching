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
- Run a cell: `PYTHONPATH=. .venv-core/bin/python -m recal.cells --method recal|prototype|aker|aker_sim|qvcache|proximity --prep <prep> --stream <name> --budget 0.05 --out <dir>`; `python -m recal.jobs` prints job lines for `simulations/approximate_hits/run_jobs.py`.
- Budget rule: every system gets the same byte budget M, a percentage of the dataset's stored bytes (N x d x bytes per coordinate), and is charged for everything it keeps in RAM to serve: vectors at their stored precision, ids, graphs and indexes, entry metadata, models and its own caches. Every result JSON carries `bytes` by structure, `M` and `within_M`.
- Baseline settings: Aker at the paper's settings (alpha_T 0.25, alpha_L 1.1, delta 0, FIFO, query-filter HNSW m 4, efConstruction 8, efSearch 16; the filter values need the shim built against the vendored library, which is the default of `sh simulations/approximate_hits/aker_real/build.sh`; a run whose FILTER line differs fails unless `--allow-unverified-filter`; `--legacy-defaults` restores the authors' bootstrap 0.9 and delta 5); Proximity is Aker's fixed-threshold mode with the 0.1 and 1 percentiles of the squared distances between distinct history queries (`--prox-percentile`); QVCache with D 0.075, alpha 0.9 and four mini-indexes, lookup and insertion per request.
- Measurement options (each behind a flag that restores the old behaviour): throughput replay aligned to recorded request ids and failing on dead workers; page cache evicted between arms (`posix_fadvise`) with randomized arm order; every composed-timing miss charged a backend latency; the quiet guard fails closed; result files written atomically and stamped with a config hash so stale results are not reused; core ranges from the machine; cache-core crash paths fixed.
- Parts that need the Linux cluster (DiskANN, the real Aker library and Proximity mode, `mincore` and `fadvise`, `/proc`-based queue and guard, MedRAG, SPACEV and TriviaQA at 10M) are implemented and mock-tested.

## Fetching data

SPACEV and SPHERE bases and Aker's request streams (Zipf 0.3, 0.6, 0.99) come from Aker's Hugging Face release (`sjoon-oh/aker`). TriviaQA, MedRAG and TripClick each have a shared Google Drive folder with a README (ask the maintainers for access; do not re-share them; TripClick's terms forbid sharing it). `datasets/MANIFEST.md` lists every folder, file, shape and transform.

```sh
python3 -m venv .venv-fetch && .venv-fetch/bin/pip install gdown
sh scripts/fetch_data.sh --list                     # what each mode fetches, no network
sh scripts/fetch_data.sh --spacev --dry-run         # actions and the free-space check, nothing downloaded
sh scripts/fetch_data.sh --spacev --triviaqa --sphere --medrag --tripclick   # or --all
sh scripts/fetch_data.sh --medrag --no-corpus       # MedRAG requests without the 74 GB corpus
sh scripts/fetch_data.sh --esci                     # ESCI for local checks (never part of --all)
```

One mode per dataset. Files are placed where the workload registry expects them (`datasets/aker/`, `datasets/medrag/`, `datasets/tripclick/`). A file expected in a Drive folder but missing from its listing stops the run before any download, the free space is checked before every download (a 10 GB floor), and each dataset is verified afterwards (`scripts/verify_data.py`). `--sphere` downloads the Hugging Face base and builds the inner-product corpus (`scripts/build_sphere_ip.py`, which checks M against the published value). MedRAG uses the 100,000-request stream and its corpus; the 10,000-request stream was made with a different rewording model and is fetched only with `--medrag-10k`, under its own names (`medrag-10k-ip` in the registry), and is not used by the experiments. Verification of the 100K stream checks the vector file (100,000 x 769 float32, last coordinate 0), the draw (Zipf 0.8, seed 0, 500 questions, counts maximum 7,602, median 92, minimum 38), the request texts (unique, question sequence equal to the draw's). Results record the request file name and SHA-256 (`data_files`).

## Setting up an instance

`sudo bash scripts/ec2_bootstrap.sh` sets up a fresh Ubuntu 24.04 x86_64 instance in steps (`--dry-run` prints them; `--only` and `--skip` select steps): the harness toolchain and PostgreSQL 16 with pgvector 0.8.0 (`harness/deploy/setup_instance.sh`), the AIO limit, `.venv-core` (Python 3.11) and `.venv-ann` from `envs/`, the ReCal and cache cores, the vendored Aker library in standard and Proximity modes against the pinned Boost and FAISS, links from `datasets/` and `simulations/approximate_hits/results` to the data volume, the data fetch and a smoke test (`scripts/smoke_cells.py`). Set `DATA_DIR` (default `/data`), `RESULTS_DIR` and `TP` to keep data, results and builds off the root disk.

## Running the evaluation

- Experiment matrix: `.venv-core/bin/python -m scripts.gen_matrix --config experiments/matrix.json --table` prints per experiment the cells, the distinct runs and the shared runs (the example configuration with MedRAG's budgets gives Q1 384, Q2 56, Q3 128 split points and 32 thread lines, Q4 93 cells), `--dry-run` counts the cells per experiment; MedRAG cells are emitted only when `medrag_budgets.json` exists (`scripts/medrag_working_set.py`) (`experiments/matrix.example.json` shows the configuration); without `--dry-run` and with `--out jobs/` it writes `jobs/q1.txt` to `q4.txt` and a manifest (`--hashes` adds the config hash of every line). Run a file with `simulations/approximate_hits/run_jobs.py jobs/q1.txt --workers N`. Each distinct run is emitted once: cells that are the same run in several experiments (same system, workload, stream, budget, mode, backend, T, seed and options; the output path, `--tag` and the logging and thread options are not part of the identity, see `recal/matrix_hash.py`) share one job line, and `matrix_runs.json` maps every experiment cell to the run that produces it. Q1: hit rate at matched recall against the baselines (T sweep, Proximity A and B, QVCache at the paper's deviation factor). Q2: ablations of ReCal. Q3: memory split and throughput. Q4: robustness.
- History: `--history-frac 0.3` (all systems): the first 30% of a stream is history, every system builds from it and only the rest is scored. Results record `history_len`, `first_scored` and `n_scored`.
- Per-request records: every run writes `<method><tag>_records.npz` next to its result JSON (path, recall, score, lambda, entries read and contributing, candidate recall, backend searches, vector re-reads, cache-path time) and ReCal adds a per-construction table. `recal/analysis/` derives every figure input from the records only: `rates`, `by_k`, `guarantee_table`, `hit_composition`, `bursts`, `hit_rate_at_recall`.
- QVCache: the authors' artifact (`https://github.com/qvcache/qvcache-vldb26`, commit pinned in `harness/deploy/pins.env`, built natively against `.venv-ann` by `scripts/ec2_bootstrap.sh`) runs through `recal/qvcache_driver.py` (`recal.cells --method qvcache`) with a backend adapter that reads our stored answers (`recal/qvcache_backends.py`; every `fetch_vectors_by_ids` call counts as vector re-reads). Settings are the artifact's defaults except the deviation factor (the paper: 0.25 for SPACEV, 0.075 otherwise, written in every cell) and the cache size, which is set so a full cache costs M; each difference is listed in the result. `scripts/validate_qvcache.py` checks the setup against the artifact's own benchmark. The byte model (772 bytes per 128-d float32 point at R 64) is checked against real memory by `scripts/measure_qvcache_memory.py`, which fits resident bytes per point and fixed overhead at several sizes and, when it differs from the model by more than 5%, writes measured constants to `recal/data/qvcache_memory.json`; every QVCache result records the source (`memory_model`). The artifact's validation sample, siftsmall, is not in its repository: `scripts/fetch_qvcache_validation_data.sh` fetches it from the texmex source and converts it with the artifact's own tools, for validating QVCache only (SIFT is not an experiment dataset). The earlier simulators stay as `--method qvcache_sim` and aker_sim, cross-checks only.
- Equal bytes: `--coord-bytes` is the stored precision (1 for SPACEV); real Aker and Proximity are sized from the library's own counts (`--equal-bytes` on `eval_aker_real.py`) and every result carries `bytes`, `M` and `within_M`.
- Backend answers and ground truth: `python -m scripts.precompute_answers --backend pgvector|diskann|exact ...` (answers to depth 7 x K_max, exact ground truth to depth 200 with `--gt-out`).
- Q3: `recal/system_benefit/` has the split budgets, node cache sizing and the best split per result cache (`best_split.py`); `scripts/node_cache_curve.py` and `scripts/depth_latency_pgvector.py` measure the backend and are not part of the matrix. A split point is a hit-rate run at the result cache's share of the budget plus a throughput arm at 64 clients with the node cache at the other share (`throughput_replay.py --arm name=<run>,nodes=N`); the thread lines replay each result cache at its best split for 1 to 64 clients after the sweep.
- Update runs are independent jobs, one per (workload, mode, rate, system, seed): each reads its own prep folder (`scripts/convert_streams.py updates --out`) and the DSN of its own PostgreSQL clone in `$PG_CLONES/<job id>.dsn` (`--pg-dsn @file`); no state is shared between jobs. Every system (ReCal, Aker, Proximity) answers its misses with live searches on that clone and applies the schedule's inserts and deletes to it in order; Aker and Proximity do so through a backend adapter in `eval_aker_real.py` (the library only sees miss answers and data-change events). `--precomputed-answers` keeps the old precomputed-and-filtered path for tests. QVCache is not in the update runs: the artifact has no update path and the paper leaves updates to the backend.
- Q4: `recal/robustness.py` (burst mask; bursts are derived from existing records by `recal/analysis/bursts.py`), `scripts/make_shift_streams.py` (abrupt, gradual and correlated shifts over 64 k-means clusters), `scripts/convert_streams.py` (turns a shift stream or an update schedule into the prep folder every driver reads, with the update files in the formats of ReCal and the real Aker driver; request vectors come from the existing official vectors, no encoder runs; cluster runs need FAISS and refuse without it), `scripts/make_update_schedule.py` and `--updates` (inserts and deletes with live ground truth).


## Running on many instances

One instance prepares each dataset, the run instances map shards of the job list, one instance reduces.

```
make prepare DATASET=spacev          # on a prepare instance: data, indexes, answer tables, preps -> store
make shards                          # jobs/shards/<workload>.json (SHARD_BY=question|system, --question/--system/--workload filters)
make run SHARD=spacev-0.6            # on a run instance: pull artifacts, run the shard, push results
make q3 WORKLOAD=tripclick SYSTEM=recal   # the jobs of one question, filtered
make reduce                          # pull results, check every run, write figures_data/*.csv
make status                          # done, missing, failed per question and system
```

The targets call `scripts/gen_matrix.py`, `scripts/store.py`, `scripts/run_shard.py` and `scripts/reduce.py`; the cell
logic is in `recal/matrix.py` only. `MATRIX` selects the configuration (default the example), `JOBS` the job directory.

- Store: `RECAL_STORE=s3://bucket/prefix` (or `RECAL_S3_BUCKET` and `RECAL_S3_PREFIX`; credentials from the instance
  role) or a directory for local use. Layout `artifacts/<dataset>/<artifact>` with `manifest.json` (path, size, sha256,
  command, code commit per artifact) and `results/<question>/<run hash>/`. Every object has a checksum sidecar that is
  verified on pull; transfers run in parallel and skip objects that match (an interrupted push or pull resumes).
  `python -m scripts.store push|pull|ls`. Every result records instance type, CPU model, cores, memory, disk type and
  the code commit (`environment`).
- Prepare stages per dataset (`recal/prepare.py`, `--dry-run` lists the commands): fetch and verify the data; request
  streams and exact answers to depth 700 with exact ground truth to depth 200; the DiskANN index at recall@10 0.95 with a
  search list of 700 (7 x K_max) and its answer tables; the pgvector database for the update workloads (parallel HNSW build,
  `maintenance_work_mem` sized by the harness, workers from the core count); shift streams (k-means seed 0 once) and update
  schedules converted once; the MedRAG working set (`medrag_budgets.json`); the QVCache memory measurement. TripClick-derived
  artifacts are not uploaded unless `--upload-restricted` is given. The stage commands for the real datasets need the cluster
  and have not been run here.
- Shards: Q3 jobs of a workload stay in one shard. Hit-rate and update runs may run concurrently; throughput jobs (split
  sweep, thread lines) are exclusive and run alone, one after the other, after the best split of each result cache is
  chosen. A shard pulls its artifacts and the results of other shards it reads, skips jobs whose result exists with the
  same config hash, and pushes results after every stage.
- Instances (estimates, to be checked on the first run): `scripts/ec2_bootstrap.sh --role prepare` fetches data and builds;
  `--role run` skips data and index builds (it installs boto3 and pulls artifacts from the store).

| Dataset | Base vectors | DiskANN index | pgvector (table + HNSW) | Answer tables | Prepare instance | Run instance |
|---|---|---|---|---|---|---|
| SPACEV (10M x 100, int8) | 1 GB | about 4 GB | about 16 GB | about 1.4 GB | 64 GB RAM, 200 GB disk | 32 GB RAM, 100 GB disk |
| TriviaQA (SPHERE, 10M x 769, float32) | 31 GB | about 34 GB | none | about 1.4 GB | 128 GB RAM, 300 GB disk | 128 GB RAM, 150 GB disk |
| MedRAG (23.9M x 769, float32) | 74 GB | about 80 GB | about 165 GB | about 0.5 GB | 256 GB RAM, 600 GB disk (pgvector HNSW build needs about 90 GB of `maintenance_work_mem`) | 192 GB RAM, 400 GB disk with the clones |
| TripClick (1.5M x 769, float32) | 4.7 GB | about 5 GB | about 10 GB | about 0.5 GB | 64 GB RAM, 100 GB disk | 32 GB RAM, 60 GB disk |

Throughput runs want the DiskANN index on local NVMe and the page cache smaller than the index. MedRAG with pgvector is
the largest instance.

### pgvector clones for update jobs

Each update job gets its own clone of the prepared database, so jobs share no state. Create the clones once per run
instance, outside the matrix: stop the prepared cluster, copy its data directory per job (`cp -a --reflink=auto` on a
filesystem with reflinks, otherwise a plain copy), start each clone on a free port with its own data directory, and write
the connection string of the clone to `$PG_CLONES/<job id>.dsn`. The job id is the update cell's name (for example
`spacev-0.99_targeted10_recal_s0`, listed in `matrix_runs.json`); the job line passes it as `--pg-dsn @$PG_CLONES/<job id>.dsn`.
