# Pre-EC2 audit

Scope: a fresh clone of main (the checkout EC2 gets), checked on a laptop with 8 GiB RAM and little disk. Every heavy step
checked free RAM and disk first. Nothing was pushed. Tests: 332 passed / 4 skipped on Python 3.12 and 333 passed / 3 skipped
on 3.10, with a real PostgreSQL 16 + pgvector 0.8.0 cluster and real Aker drivers (macOS arm64). `make q1-smoke` ran end to end
and left `git status` empty. The skipped tests need torch or CUDA.

## Fixes (one line each, with the commit)

Bugs that would have failed or silently damaged a run on the instance:

| Commit | Problem | Fix |
|---|---|---|
| dd703fa | HNSW build memory estimate used 16 bytes of element overhead; measured on pgvector 0.8.0 it is about 204. An m=16 build (spacev, sift10m, esci) with the production formula plus its 1.15 margin ran out of `maintenance_work_mem` at 95% of the tuples, then ran about 2x slower and failed the "graph no longer fits" guard after the build. | `4*dim + 32*m + 208` bytes per vector, fitted to the server notice for four shapes (see "Measurements"). SPHERE needs about 43 GB for the graph, so about 49.5 GB of `maintenance_work_mem`. |
| 60cd7ae | Setup would fail on stock Ubuntu 22.04: Boost 1.74 (< 1.80 needed), CMake 3.22 (< 3.23.1 needed by FAISS 1.8.0), libfaiss-dev 1.7.2 never compiled against; apt calls interactive; PostgreSQL on the default port next to the packaged cluster; conda on PATH. | Pinned toolchain (Boost 1.84 tarball with sha256, CMake pip wheel, FAISS v1.8.0 commit, pgvector 0.8.0 commit), non-interactive apt, own PostgreSQL cluster on port 54320, hashed lock files, driver RPATH, dry-run tests for jammy and noble. |
| 39577d2 | `make q1-generate` never exported `Q1_PGDATA`, so the PostgreSQL disk check was silently skipped; disk estimate ignored the raw source and WAL; no RAM check before the index build. | Data directory from `Q1_PGDATA` or `SHOW data_directory`; components (raw, float32, table, index, WAL allowance, other) in `need_bytes`; RAM refusal at `pg_index`. SPHERE raw is float64 (61.4 GB). |
| 0d55fe7 | GPU ground truth tile held int64 ids (8 bytes per element) that the memory model did not count: 8192 x 131200 ids alone is 8.6 GB, a likely OOM on a 24 GB card. | `fit_chunks` counts 20 bytes per tile element, resident base uploaded first, `MemoryError` when nothing fits. |
| a565163 | CUDA matmul precision not pinned (TF32 would make candidate distances too coarse for the 28-candidate margin); CPU encoding of SPHERE texts not refused. | `allow_tf32 = False`; refusal like ESCI. |
| 26e9a35 | Ground truth, answer table and result writes used `name.tmp.npz`, which matches the `chunk_*`/`block_*`/`*.npz` globs: a kill between write and rename left a phantom chunk (resume then failed with "N+1 of N chunks"), or a stale `.tmp.npz` in the results directory that `collect` would read. | Atomic helper with pid-tagged names; starting a stage removes temporaries of dead writers. SIGKILL and resume tests compare resumed arrays with an uninterrupted run. |
| c871873 | Universe, streams, tune.json, backend_recall.json, env.json, answers_first10.npz, stage markers written in place. | All go through `workloads/q1/atomic.py`. |
| 9730684 | ESCI embedding: one `np.save` at the end; the embed stage skipped when the files existed, so a crash left a truncated products file that later runs trusted; the sampled-match proof existed but was never used. | Shards of 100K texts, atomic, manifest (model, text recipe, normalization, dim, texts hash, code commit); reuse only after a sampled re-encode matches; shards of other texts are refused. |
| 05da101 | Results were reused by commit alone; shared driver inputs and the percentile cache were keyed by dataset/stream/seed only, so a rebuilt answer table, universe, or patched library was silently served from old files. | `harness/fingerprints.py`: inputs hashed at the end of generation, verified at replay start, part of `is_done`, of the shared-input and percentile cache names; result headers carry fingerprints, git tree ids of `third_party/aker` and `harness/cpp`, driver sha256, generation env, versions; drivers built from other sources are refused. |
| b6f1b0d | No download was verified. | Published sha256 (Hugging Face `x-linked-etag`, GitHub LFS pointer), git sha1, Azure Content-MD5, S3 ETag incl. multipart; trust-on-first-use sidecar for the SIFT slice; mismatches quarantined as `.corrupt`. Checked live on three small files and the published values of the big ones. |
| dd703fa | Answer-table entry point did not assert `ef_search >= list_size`; CLI `--list-size` defaulted to a stale 50; validation sampled 2000 rows and only the first-10 file. | Assertion in `[list_size, 1000]`, required `--list-size`, validation of every block (range, repeats, sorted finite distances, rows, meta), stored distances recomputed on a sample, recall on the tuning queries compared with `tune.json`. |
| dd703fa | Client sessions had no TCP keepalives; index session did not clear `statement_timeout`. | `with_keepalives(dsn)` on every pipeline connection; `SET statement_timeout = 0` in the build session. |
| e2ba8e8 | `.bin`/`.f32bin` readers did not check file size against the header; the driver read the 8-byte header without checking the file length; answer tables store int32 ids without a base-size guard. | Size validation in Python and C++, guard at 2^31 rows. Offset tests past 2^31 with a virtual 128 GB file. |
| ad52491 | `q1-bg` used `nohup` only, inherited make's jobserver (so `make -j q1-bg` blocked until the job ended), had one shared pid file and no status. | `setsid` when present, `/dev/null` stdin, jobserver flags and pipes dropped, pid and log per run, `make q1-status`, rotating generation log. |
| 2dcf06b | An unmounted `/data` after a reboot would fill the root volume. | Runtime refusal on the root volume (`Q1_ALLOW_ROOT_VOLUME=1` to accept). |
| a762209 | `tarfile.extractall(filter=)` does not exist before Python 3.10.12; one timing-sensitive test. | Use the filter only when supported (members are validated first). |
| 63d6842 | `triviaqa-encode` could fall back to downloading the tokenizer. | `--tokenizer-dir` required. |
| (this fix) | Found on the first real instance (Ubuntu 24.04): `wait_for_apt` matched process names with `pgrep -x unattended-upgr`. The kernel cuts a script's name to 15 characters, so the idle `unattended-upgrade-shutdown` daemon that runs on every Ubuntu server matched forever (the real upgrader has the same name) and setup never got past the wait. The audit missed it because `Q1_DRYRUN` skips the function, no test executed it, and a Mac has no unattended-upgrades. | Wait by lock ownership: inode of each apt/dpkg lock file looked up in `/proc/locks`, bounded by `Q1_APT_WAIT_SECONDS` (900), error names lock, pid and command line, nothing is killed or deleted; every `apt_get` waits first and passes `Acquire::Retries`/timeouts; `cloud-init status` is awaited; `sudo -n` check; timeouts on `curl` and `git fetch`; no git/pip/ssh prompts. Tests in `tests/harness/test_apt_wait.py`. |
| 084cf7b | No record of the values the driver and FAISS actually hold; peak pool was sampled every 100 requests. | Driver prints a `PARAM` line and a `FILTER` line (M, efConstruction, efSearch as held by FAISS); Python checks them against the intended settings and fails on a mismatch. Exact peak pool via a constant-time accessor (local patch 6). Library counters cross-checked with our classification (header field `counter_disagreements`, 0 in all tests). |

## Measurements (scratch PostgreSQL 16.15 with the same settings as `setup_instance.sh`)

- Bytes per tuple at which "graph no longer fits" fires: (dim 100, m 16) 1115, (128, 16) 1228, (384, 16) 2250, (768, 32) 4303; serial and parallel builds agree.
- 100K x 100 build, m 16, efConstruction 128: 7 parallel workers observed, 17 s, peak leader RSS 188 MB, index 71 MB, table 43.9 MB. At 0.97x of the fitted estimate the build spills and takes 35 s.
- WAL: `wal_level=minimal` wrote 0.4 MB for the load and 0.0 MB for the index; `replica` wrote 47 MB and 64 MB (about table and index size). The disk estimate keeps a WAL allowance of min(64 GiB, float32 base) to stay safe.
- Setup config works with `unix_socket_directories = ''` (`pg_ctl`, `psql` over 127.0.0.1).

## Cross-checks that agree

Naive loop recall equals production on ties, duplicates, short lists and padding. A naive Proximity reference (nearest stored query within the squared-L2 threshold, no eviction) agrees with the C++ driver to within 3 of 300 requests. Misses return the backend answer in stored order; hits return a stored list unchanged; no-locality streams have zero equality hits; the uniform stream is exactly the base queries; exact hits score recall 1.0; ground-truth distances and answer distances equal a brute-force recomputation; the same cell is byte-identical across reruns, `OMP_NUM_THREADS` 1 vs 4 and different `PYTHONHASHSEED`; the window is requests 30000..99999 and baselines replay all 100K. Linked repeats are `sim-hit` in the library, matching our rule that only exact-id entries are equality.

## Needs your decision (not changed)

1. Ground truth top-k merge. The GPU path keeps a running top-(100+28) and recomputes exact distances on the CPU. With heavy exact ties at the 100th distance (SPACEV is int8, so ties are common) which tied ids are kept can differ between devices and chunkings; distances are always exact and tests show no duplicate or missing ids. A two-stage per-chunk top-k would lower GPU memory but changes which tied ids are kept. I left it alone.
2. Nothing else touched equality classification, thresholds, answer-table contents, universe or stream construction.

## Risks remaining

- The first Linux build of Boost 1.84, FAISS 1.8.0 and the drivers has not run; the scripts were only dry-run tested with fake system tools. The first minutes on EC2 are for this (see `EC2_FIRST_RUN.md`).
- SPHERE-scale memory: the fitted graph size is exact per tuple, but build time at 10M x 768 is unmeasured; the estimate says about 49.5 GB of `maintenance_work_mem` plus 4 GiB.
- Binary checksum of the multipart ETag of the DPR archive is implemented with an 8 MiB part-size assumption and tested only on synthetic files; if it mismatches on the real archive, check the part size before suspecting the file.
- Unverified WAL behavior at full size under `wal_level=minimal`.
- The apt lock wait reads `/proc/locks`; its real-kernel test (`test_real_kernel_lock_is_seen_and_released`) only runs on Linux, so it is first exercised on the instance (see `EC2_FIRST_RUN.md`).
- `harness/status.py` treats a pid file as running if the pid exists; after a reboot a reused pid would show as running (display only).

## Could not be verified here

Ubuntu 22.04/24.04 package behavior and the source builds; Docker (not available); any CUDA path (TF32 pin, tile model, `torch.zeros(1).cuda()` check, driver minimum); the real TriviaQA encode with no network (no torch here; the CLI now cannot reach the hub); full-size downloads and builds; the DPR tarball ETag check on the real file; SSH disconnect behavior of `q1-bg` with `setsid` (tested with `nohup` and SIGHUP on macOS); the CI workflow (skipped on request).
