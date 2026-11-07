# Harness (Q1)

Aker and Proximity (Aker's fixed-threshold mode) replayed over a precomputed pgvector answer
table. `harness/DECISIONS.md` lists decisions, deviations and findings; `harness/aker/DIFFERENCES.md` lists
every difference from the paper; `third_party/aker/PATCHES.md` lists the local changes to the library.

## Run book (EC2, Ubuntu 22.04 or 24.04, Linux x86_64, one CUDA GPU)

Suggested instance (a recommendation, not measured): one GPU instance with about 192 GB RAM
(for example g5.12xlarge) and about 1.5 TB of gp3. Clone the repo at a commit and keep the
tree clean (every q1 target refuses a dirty tree).

    make q1-setup                    # packages, PostgreSQL 16 + pgvector v0.8.0, FAISS, venvs, drivers
    make q1-pg-start                 # if the cluster is not running
    make q1-smoke                    # whole stack on a tiny synthetic dataset (about a minute)

    make q1-generate DATASET=esci    # then spacev, sift10m, sphere (or DATASET=all)
    make q1-run-aker-b1              # one system config over every dataset and stream
    make q1-run-proximity-p0.1-b5    # ... and the 9 shortcuts:
                                     #   q1-run-aker-b{1,5,10}
                                     #   q1-run-proximity-p0.1-b{1,5,10}
                                     #   q1-run-proximity-p1-b{1,5,10}
    make q1-run-all                  # all nine, in order
    make q1-collect                  # table + summary.csv in the results directory
    make q1-fidelity                 # Aker on Aker's released SPACEV stream (report only)
    make q1-snapshot SNAPSHOT=name   # tar of the stopped PostgreSQL data directory
    make q1-restore SNAPSHOT=name

Run long jobs detached: `make q1-bg TARGET="q1-generate DATASET=esci"` (log in `$Q1_DATA/logs/`).
`make -j 9 q1-run-all` runs the nine configs in parallel. They read the precomputed table and do not need
PostgreSQL, so `make q1-pg-stop` before them frees memory for the page cache.

Logs: `$Q1_DATA/<dataset>/.logs/generate.log` (all stages, with progress and ETA),
`answer_table_<i>.log` per shard, `env.json` (versions, commit), `.failed/<stage>.json` after a failure.
A failed stage stops `make` with the stage name and the log tail. Results: `$Q1_DATA/results/<commit>/`.

Variables: `Q1_DATA` (default /data/q1), `Q1_LIST_SIZE` (L, default 50), `Q1_ALLOW_SLOW=1` (accept CPU ground
truth or CPU ESCI embedding), `Q1_PG_DSN`, `Q1_PG_WORKERS` (parallel answer-table processes), `Q1_DEVICE` (auto, cuda, numpy), `DATASETS` (restrict a run to some datasets),
`Q1_DRIVERS`, `GT_QUERY_CHUNK`/`GT_BASE_CHUNK`.

`q1-generate` runs these stages in order and skips finished ones (markers in
`$Q1_DATA/<dataset>/.done/`): fetch, embed, to_f32, inputs, groundtruth, pg_load, pg_index,
tune_ef (writes tune.json), answer_table, backend_recall (writes backend_recall.json).
`make q1-stage DATASET=esci STAGE=groundtruth FORCE=1` redoes one stage and all later ones.
Runs are resumable: finished result cells are skipped, answer-table and ground-truth blocks are
written atomically.

Layout of a prepared dataset directory: `raw/`, `base.f32bin`, `official.f32bin`,
`universe.npz`, `streams_seed0.npz`, `gt/`, `answers/`, `tune.json`, `backend_recall.json`.

## Local smoke build (macOS, `brew install cmake faiss boost libomp postgresql@16`)

    make build-replay
    make q1-smoke Q1_ALLOW_DIRTY=1 Q1_DATA=<tmp> Q1_PG_DSN="host=127.0.0.1 port=<p> user=<u>"

Tests: `make test-q1` (`PGVECTOR_TEST_DSN` enables the real PostgreSQL tests).
