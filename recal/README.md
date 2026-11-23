# ReCal

ReCal is a recall-certified result cache for top-K vector search. It has two tiers: a mini cache for exact repeats and a main cache of spheres around past queries. A main-cache answer is served only when a calibrated score says the expected recall of the answer is at least the target T, with confidence 1 - delta. The earlier cache implementation is kept under `simulations/approximate_hits/` for comparison.

## Layout

| Path | Contents |
|---|---|
| `recal/core/recal_core.cpp` | C++ (pybind11) storage and serving core: data pool, shell pool, reverse map, entries, center index (HNSW), mini cache, serving order, byte accounting, deletes, sweep, batched inserts. Build: `sh recal/core/build.sh` (needs `.venv-core`, Python 3.11) |
| `recal/params.py` | Every parameter with its default |
| `recal/backend.py`, `recal/stored.py` | Backends: `ExactBackend` (live exact search, data changes), `StoredAnswers` (exact top-R tables, DiskANN `diskann.npz` rows, `from_pgvector` for the pgvector answer table) |
| `recal/history.py` | History log and hash log on disk (memory-mapped ring buffers), reuse distances |
| `recal/policy.py` | Hit policy: estimator, thresholds, calibration window, certification, status |
| `recal/build.py` | Construction: labels, roles, proposals, selection, shells, m, training, seeding |
| `recal/updates.py`, `recal/signals.py` | Data updates and rebuild signals |
| `recal/system.py` | The serving loop (`ReCal.serve`), construction and swap, reports |
| `recal/sim.py`, `recal/cells.py`, `recal/jobs.py`, `recal/accounting.py` | Experiment cells, equal-byte budgets for every system, job lines |
| `tests/recal/` | Tests; `tests/recal/reference/prototype.py` holds reference functions of the earlier cache implementation, used as the oracle |

## Components

| Part of ReCal | Where |
|---|---|
| Entries (center, K, size, response list, radius, shell ranks), data pool, shared 4-bit shell pool, reverse map from database id to slot, center index | `recal_core.cpp`: `add_entry`, `add_shell`, `finish_build`; tests in `test_core.py` |
| Exact responses when a ball contains the query | `test_containment_gives_exact_response` |
| Mini cache: hash of the query bytes to the backend's ranked answer, LRU | `recal_core.cpp`: `mini_*`, `hash_vector`; mini tests in `test_core.py` |
| Memory accounting by structure | `Core.bytes_by_structure()`, `ReCal.bytes_by_structure()` (headroom is consumed by inserts); the history and hash logs are files |
| History log and hash log | `history.py` |
| Cold start (the mini cache runs at its cap until the first construction) | `system.py` (`cold_start`) |
| Labels from the backend (fetch the unlabeled, refetch all if the data changed since the oldest response) | `build.fetch_labels` |
| Roles of the history (candidate, selection, training, calibration) | `build.split_roles` (by position, or `split_by_query`) |
| Proposals (one 7K search per center and per old center, five sizes) | `build.propose` |
| Greedy selection by hits per byte | `build.hits_from_hist`, `build.Greedy` |
| Mini cache size from reuse distances | `build.mini_cache_size`, `history.reuse_distances` |
| Construction and atomic swap (mini cache carried over) | `build.build`, `system.construct` |
| Serving order, features (offset, size, intruder rate), candidate union | `Core.lookup` |
| Entries probed per query (m) | `Core.entries_needed_batch`, `build.build` (95th percentile, capped) |
| Recall estimator | `policy.logistic_model`, `Core.set_model` |
| Hit policy: thresholds, weighted window, certification, recalibration, status | `policy.py`, `system._label`, `system.status_report` |
| Deletes, sweep, batched inserts, displacement | `Core.delete_id`, `sweep`, `insert_batch`; `updates.py` |
| Rebuild signals | `signals.py` |
| Responses | ranked ids with distances from every path; recall is always against the exact top-K; the deepest fetch (7K) needs a backend tuned for that depth |

## Parameters (defaults in `params.py`)

K 10, k_max 100, T 0.9, delta 0.05, audit rate a 0.1, history H 30,000 main-path requests, cold start 10,000, roles 20/40/20/20, entry sizes 1K to 5K, shell ranks jK+1 to (j+2)K, deepest fetch 7K, shell quantization 4 bits, m = 95th percentile capped at the number of entries, 100 threshold quantiles, minimum effective sample size 100, window min(H, 200/(a h0)), recalibration on every labeled query, mini cache min(R95, 10% of M) with at least 256 entries, headroom 2% of M, sweep at 5% dead slots, insert batches of max(100, 0.1% N) with 16 centers per round and 5 rounds, shrinking pool 80%, frozen inserts 5%, score drop windows 0.2H (floor 2,000) checked every 1,000 from 0.4H at 3 standard errors, outage 90% of the last 0.2H, minimum interval between rebuilds 0.2H.

## Running

- Python API: `ReCal(backend, Params(), M_bytes, workdir)`, then `serve(query, K)`; `construct()`, `insert(ids, vectors)`, `delete(ids)`, `status_report()`, `bytes_by_structure()`.
- Deployable mode (default): candidates from the m nearest entries. Idealized mode: `Params(idealized=True)` or `--idealized`, candidates from all entries, no m cap.
- Cells: `PYTHONPATH=. .venv-core/bin/python -m recal.cells --method recal --prep <prep> --stream random --budget 0.05 --out <dir>`; methods `recal`, `prototype`, `aker`, `qvcache` (simulators), `proximity` (needs the cluster); `prototype` is the earlier cache implementation. `python -m recal.jobs` prints job lines for `simulations/approximate_hits/run_jobs.py`.
- Local checks on ESCI 50K: `scripts/recal_esci_static.py` (one construction against the earlier implementation), `scripts/recal_stream.py` (end to end with per-window recall of the hits), `scripts/recal_updates_stream.py` (inserts and deletes through rebuilds), `scripts/make_local_prep.py` (prep folder).
- Tests: `.venv-core/bin/python -m pytest tests/recal`.

## Budget rule

M is a percentage of the dataset's stored bytes. Every system is charged for everything it keeps in RAM to serve (`accounting.py`, `bytes_by_structure`), and every result JSON carries `bytes`, `M` and `within_M`.
