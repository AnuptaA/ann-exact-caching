# Harness (Q1)

Aker and Proximity (Aker's fixed-threshold mode) replayed over a precomputed pgvector answer
table. Layout of a prepared dataset directory: `base.f32bin`, `official.f32bin`, `universe.npz`,
`streams_seed0.npz`, `answers/`, `gt/`.

Local smoke build (macOS, needs `brew install cmake faiss boost libomp`):

    cmake -S harness/cpp -B build/replay -DBOOST_ROOT=$(brew --prefix boost) \
        -DFAISS_ROOT=$(brew --prefix faiss) -DLIBOMP_ROOT=$(brew --prefix libomp)
    cmake --build build/replay -j 4

Linux x86_64 instance: `harness/deploy/deploy.sh <repo> <commit> <dir>` (refuses a dirty tree,
runs preflight, builds). Pipeline per dataset:

    python -m harness prepare-inputs ...   # base, official, universe, streams
    python -m harness groundtruth ...      # exact top-100 for every request vector
    python -m harness load-base / build-index ...   # PostgreSQL 16 + pgvector 0.8.0
    python -m harness tune-ef ...          # ef_search = max(smallest with recall >= 0.9, list size)
    python -m harness answer-table ...     # backend answers, resumable and shardable
    python -m harness run --shard i/n ...  # the grid; completed cells are skipped

`harness/aker/DIFFERENCES.md` lists every difference between this setup and the paper.
