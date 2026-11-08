# First 30 minutes on EC2

Run from a clean checkout of the audited commit. Do not edit anything on the instance. Stop at the first failed check.

## 0. Before anything (2 min)

```
git clone <repo> thesis && cd thesis && git checkout --detach <commit> && git status --short   # prints nothing
lsblk; df -h /data                                                                              # /data is its own volume, >= 400 GB free
nvidia-smi                                                                                      # a GPU and a driver; note the version
```

Go only if `/data` is a separate mounted volume. Do not set `Q1_ALLOW_ROOT_VOLUME=1` to get past this.

Check the apt lock wait against the real kernel once (it must fail and name the python pid, then pass after the holder is gone):

```
sudo python3 -c 'import fcntl,time;f=open("/var/lib/dpkg/lock-frontend","a");fcntl.lockf(f,fcntl.LOCK_EX);time.sleep(60)' &
Q1_APT_WAIT_SECONDS=5 bash -c '. harness/deploy/lib.sh; wait_for_apt'    # fails: .../lock-frontend held by pid N: python3 -c ...
sudo kill N                                                                # N from the message, or wait 60 s
Q1_APT_WAIT_SECONDS=5 bash -c '. harness/deploy/lib.sh; wait_for_apt' && echo free
```

## 1. Setup (10-20 min, mostly compiling)

```
make q1-setup 2>&1 | tee ~/setup.log
```

Setup first waits (at most 10 min for cloud-init, 15 min for the apt/dpkg locks) and then proceeds; if a lock stays held it stops and prints the lock file, the holder's pid and its command line. It never kills anything and never touches a lock file. An idle `unattended-upgrade-shutdown` daemon does not count as a holder.

Setup can be re-run: every step skips what is already done, and `pg.sh start` accepts a cluster that already runs. `tail -f ~/setup.log` never exits by itself; the run is over when the log ends with `setup done` or `setup FAILED: <steps>` (then press Ctrl-C). If `pg_start` fails, the summary note and the log show the end of the server log; the full file is `/data/q1/pgdata.log` (`/data/q1/pgdata` is the data directory, `pg_ctl -D /data/q1/pgdata status` tells whether it runs).

Setup runs every independent step and ends with a summary table (PASS, FAIL, SKIPPED, N/A, NOT RUN per step); the GPU check is the last step, so a missing GPU never hides the builds. Exit code 0 needs every step to pass.

Watch for: Boost 1.84 build (only on 22.04), CMake wheel, FAISS v1.8.0 build, PostgreSQL 16 + pgvector 0.8.0, both drivers, then the `preflight` call. Expected end: `preflight ok`. `build/replay/TOOLCHAIN.txt` lists commit, cmake, gcc, boost, faiss. `git status --short` must still print nothing.

Checks:

```
ls build/replay/ak_replay_standard build/replay/ak_replay_proximity
ldd build/replay/ak_replay_standard | grep -i "not found"                    # prints nothing
.venv/bin/python -m harness preflight --require-gpu --data-dir /data/q1 --dsn "host=127.0.0.1 port=54320 user=postgres"   # preflight ok
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.6.0..., True
psql "host=127.0.0.1 port=54320 user=postgres" -Atc "show wal_level; show max_parallel_maintenance_workers; select default_version from pg_available_extensions where name='vector'"
```

Expected: `minimal`, a number between 1 and 7, `0.8.0`.

## 1b. CPU-only dry run (cheap instance or CI, before paying for a GPU)

Verifies the Linux build, PostgreSQL 16, pgvector 0.8.0, both drivers and the smoke test without a GPU, on a machine too
small for the real datasets:

```
Q1_ALLOW_NO_GPU=1 Q1_ALLOW_SHORTFALL=1 Q1_MIN_DATA_GB=10 Q1_SETUP_DATASETS=tiny make q1-setup 2>&1 | tee ~/setup.log
make q1-smoke 2>&1 | tee ~/smoke.log
git status --short                                   # prints nothing
```

Expected: every row of the summary is PASS except `gpu` (`WARN`, "no NVIDIA driver or GPU found"; a driver too old for torch
gives a different message) and `preflight` (`WARN`, with the disk and RAM numbers), and the exit code is 0. A loud "DRY-RUN MODE"
banner is printed. Without the two flags the same run exits 1 with `gpu FAIL` and/or `preflight FAIL`, and every other step
still passes. The flags only downgrade the setup checks: ground truth, ESCI/SPHERE encoding and `--device cuda` still refuse
to run without a GPU, `make q1-preflight` stays strict, and nothing real can be generated on that machine. Only the exact value
`1` enables a flag. `Q1_SETUP_DATASETS` takes `tiny`, one dataset name, several names, or `none`.

`build/replay/TOOLCHAIN.txt` must list the BLAS the drivers link, for example `blas_vendor OpenBLAS` and a
`blas_lib /usr/lib/x86_64-linux-gnu/libopenblasp-r0.3.xx.so libopenblas0-pthread <version>`, plus the OpenMP library. The driver
build fails (instead of printing 40 undefined-reference lines) if no `libopenblas` is installed or if it links a different BLAS
than FAISS was built with (`/opt/q1/faiss/BLAS_INFO` records FAISS's). An existing FAISS install is not rebuilt: its BLAS is
read from its build tree.

Without a machine: Actions tab, "Q1 Linux build", Run workflow on the branch (`workflow_dispatch` only, Ubuntu 22.04 and 24.04
x86_64). It runs `setup_instance.sh --ci`, the `ldd` check, prints `TOOLCHAIN.txt`, runs `make q1-smoke`, and requires an
empty `git status --short`.

## 2. Smoke test (1 min)

```
make q1-smoke 2>&1 | tee ~/smoke.log
```

Expected: ten `stage ... done` lines for `tiny`, then a collect table with eight rows, no `FAILED`. Then `git status --short` prints nothing.

## 3. Look at one driver log (1 min)

```
head -3 "$(find /data/q1/work -name driver.log | head -1)"
```

The driver logs live under `$Q1_WORK` (default `/data/q1/work/<cell>/driver.log`). The log must contain `FILTER faiss_M=4 efConstruction=8 efSearch=16` and a `PARAM ...` line with `alpha_tighten=0.25 alpha_loosen=1.1 top_delta=0 filter_m=4 filter_ef_construction=8 filter_ef_search=16`. The pipeline already fails on any mismatch; this is the human check.

## 4. First ESCI generation

```
make q1-bg TARGET="q1-generate DATASET=esci"
make q1-status
tail -f /data/q1/logs/<newest>.log
```

Expected order: `fetch` (verified against the published sha256 lines), `embed` (shards, ETA), `to_f32`, `inputs`, `groundtruth`, `pg_load`, `pg_index`, `tune_ef`, `answer_table`, `backend_recall`. `make q1-status` lists the stages done and the next one.

### Go / no-go for the first ESCI generation

Go when all hold:

- `fetch`: each file logs `verified against the published value` (ESCI's two parquet files publish a sha256 through their LFS pointers).
- `embed`: shards appear in `/data/q1/esci/raw/esci_products.npy.shards`; the manifest (`esci_products.npy.manifest.json`) lists the model, recipe, and commit. GPU speed: note minutes per 100K shard; the ETA line gives the total. If the ETA is over about 1 hour, stop and check the GPU is used.
- `groundtruth`: no `chunks reduced` warning, or a small reduction only; ETA is sensible.
- `pg_index`: for esci (m 16, dim 384) the log shows `maintenance_work_mem` of about 4.7 GB (2256 bytes per vector x 1.8M x 1.15); no `server notice`; `index built in X min`.
- `tune_ef`: `ef_search` between 200 and 400 and recall at that ef at least 0.9 (the stage fails otherwise).
- `answer_table`: all shards exit 0; validation passes (the stage fails on any bad block); `answers_first10.npz` exists.
- `backend_recall.json` mean recall plausible (0.9 to 1.0); `fingerprints.json` exists; `make q1-status` shows all ten stages done.
- Resource lines: disk free stays above 100 GB; RAM available never near zero during `pg_index`.

No-go: any `FAILED`, a `server notice`, ef_search above 400, recall below 0.9, unexplained slowdowns (index build much slower than the 17 s per 100K x 100 measured here suggests when scaled, i.e. if it is more than 3x the linear extrapolation).

## 5. First Aker run

```
make q1-run SYSTEM=aker SETTING=paper BUDGET=1 DATASETS=esci
make q1-collect
```

### Go / no-go for the first Aker run

Go when all hold, per cell (they are in the result header; `python -m harness collect` prints the table):

- Startup values: `PARAM`/`FILTER` as in step 3; `pool_size` equals 1% of N (esci: 18,149) exactly.
- `peak_pool_vectors` at most `pool_size`; `counter_disagreements` is 0.
- Hit rate and recall in a sane range: recall on misses equals the backend recall; delivered recall at most about 1.0; Zipf 0.99 hit rate above uniform's.
- Runtime within the extrapolation from the tiny run (about 0.3 s per 2000 requests there; real stream is 100,000 requests with a pool of N/100).
- `git status --short` still empty, and results appear under `/data/q1/results/<short commit>/`.

No-go: a fingerprint mismatch message (regenerate the named input), a missing `PARAM` line (stale driver), `counter_disagreements` above 0, or peak pool above `pool_size`.

Then run the other eight configs (`make -j 9 q1-run-all` or the nine shortcuts), and the other three datasets in the order esci, spacev, sift10m, sphere. SPHERE needs about 49.5 GB for the index build plus 4 GiB; `make q1-preflight` refuses a machine with less.
