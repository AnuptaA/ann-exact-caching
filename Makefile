VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

K ?=
_K_FLAG = $(if $(K),--K $(K),)

.PHONY: all help
.PHONY: synthetic-generate synthetic-run synthetic-analyze
.PHONY: esci-prepare esci-run esci-analyze
.PHONY: esci-population-prepare esci-population-run esci-population-run-all esci-population-analyze esci-population
.PHONY: esci-population-run-K esci-population-run-all-K
.PHONY: esci-alpha-microbenchmark esci-k-microbenchmark
.PHONY: population
.PHONY: characterize
.PHONY: test \
        test-utils test-distance-metrics test-kv-cache test-lemmas test-main-memory \
        test-perturbation test-simulate \
        test-range-search test-population test-selection \
        test-synthetic test-generator test-e2e-synthetic test-analysis \
        test-esci test-esci-edge-cases test-esci-e2e \
        test-esci-population-e2e test-esci-microbenchmark test-esci-microbenchmark-k \
        test-population-all
.PHONY: count-datasets
.PHONY: clobber clobber-synthetic clobber-esci clean-cache

all: help

help:
	@echo "Synthetic:  synthetic-generate | synthetic-run | synthetic-analyze"
	@echo "ESCI:       esci-prepare | esci-run | esci-analyze"
	@echo "            esci-population-prepare | esci-population-run | esci-population-analyze"
	@echo "            esci-population  (all three in sequence)"
	@echo "            esci-population-run K=<n>  (quick run with K override)"
	@echo "            esci-population-run-K      (quick run for all K=[1,5,10,20])"
	@echo "            esci-population-run-all-K  (full run for all K=[1,5,10,20])"
	@echo "Combined:   population  (esci-population)"
	@echo "                esci-alpha-microbenchmark | esci-k-microbenchmark"
	@echo "All:        analyze  (runs all workloads + collects results)"
	@echo "Characterize: characterize (all workloads, cross-workload plots)"
	@echo "Tests:      test | test-utils | test-synthetic | test-esci"
	@echo "            test-population-all  (all population tests)"
	@echo "            utils:     test-distance-metrics test-kv-cache test-lemmas"
	@echo "                       test-main-memory test-perturbation test-simulate"
	@echo "                       test-range-search test-population test-selection"
	@echo "            synthetic: test-generator test-e2e-synthetic test-analysis"
	@echo "            esci:      test-esci-edge-cases test-esci-e2e"
	@echo "                       test-esci-population-e2e"
	@echo "                       test-esci-microbenchmark test-esci-microbenchmark-k"
	@echo "Utilities:  count-datasets"
	@echo "Cleanup:    clobber[-synthetic|-esci]"

# ── Synthetic workflow ────────────────────────────────────────────────────────

synthetic-generate:
	$(PYTHON) datasets/synthetic/generator.py --test-set all

synthetic-run:
	$(PYTHON) simulations/synthetic/run_synthetic.py

synthetic-analyze:
	$(PYTHON) simulations/synthetic/analyze_synthetic.py
	$(PYTHON) simulations/synthetic/compile_pdf.py

# ── ESCI workflow ─────────────────────────────────────────────────────────────

esci-prepare:
	$(PYTHON) datasets/esci/prepare_benchmark.py --all

esci-run:
	$(PYTHON) simulations/esci/run_esci.py

esci-analyze:
	$(PYTHON) simulations/esci/analyze_esci.py
	$(PYTHON) simulations/esci/compile_pdf.py

# ── ESCI population workflow ──────────────────────────────────────────────────

datasets/esci/.population_prepared:
	$(PYTHON) datasets/esci/prepare_benchmark.py --all
	$(PYTHON) datasets/esci/prepare_benchmark.py --extend-population
	touch $@

esci-population-prepare: datasets/esci/.population_prepared

esci-population-run: datasets/esci/.population_prepared
	$(PYTHON) simulations/esci/run_population_esci.py --quick $(_K_FLAG)

esci-population-run-all: datasets/esci/.population_prepared
	$(PYTHON) simulations/esci/run_population_esci.py $(_K_FLAG)

esci-population-run-K: datasets/esci/.population_prepared
	for k in 1 5 10 20; do \
		$(PYTHON) simulations/esci/run_population_esci.py --quick --K $$k; \
	done

esci-population-run-all-K: datasets/esci/.population_prepared
	for k in 1 5 10 20; do \
		$(PYTHON) simulations/esci/run_population_esci.py --K $$k; \
	done

esci-population-analyze:
	$(PYTHON) simulations/esci/analyze_population_esci.py

esci-population: esci-population-prepare esci-population-run esci-population-analyze

esci-alpha-microbenchmark: datasets/esci/.population_prepared
	$(PYTHON) simulations/esci/microbenchmark_candidates.py

esci-k-microbenchmark: datasets/esci/.population_prepared
	$(PYTHON) simulations/esci/microbenchmark_k.py

# ── Combined population ───────────────────────────────────────────────────────

population: esci-population

# ── Combined analysis ─────────────────────────────────────────────────────────

analyze: synthetic-analyze esci-analyze
	$(PYTHON) simulations/collect_results.py

# ── Characterization ──────────────────────────────────────────────────────────

characterize:
	$(PYTHON) simulations/characterize.py

# ── Tests ─────────────────────────────────────────────────────────────────────

test: test-utils test-synthetic test-esci

# utils tests
test-utils: test-distance-metrics test-kv-cache test-lemmas test-main-memory \
            test-perturbation test-simulate \
            test-range-search test-population test-selection

test-distance-metrics:
	$(PYTHON) tests/utils/test_distance_metrics.py

test-kv-cache:
	$(PYTHON) tests/utils/test_kv_cache.py

test-lemmas:
	$(PYTHON) tests/utils/test_lemmas.py

test-main-memory:
	$(PYTHON) tests/utils/test_main_memory.py

test-perturbation:
	$(PYTHON) tests/utils/test_perturbation.py

test-simulate:
	$(PYTHON) tests/utils/test_simulate.py

test-range-search:
	$(PYTHON) tests/utils/test_range_search.py

test-population:
	$(PYTHON) tests/utils/test_population.py

test-selection:
	$(PYTHON) tests/utils/test_selection.py

# synthetic tests
test-synthetic: test-generator test-e2e-synthetic test-analysis

test-generator:
	$(PYTHON) tests/synthetic/test_generator.py

test-e2e-synthetic:
	$(PYTHON) tests/synthetic/test_e2e_mini.py

test-analysis:
	$(PYTHON) tests/synthetic/test_analysis.py

# esci tests
test-esci: test-esci-edge-cases test-esci-e2e test-esci-population-e2e test-esci-microbenchmark test-esci-microbenchmark-k

test-esci-edge-cases:
	$(PYTHON) tests/esci/test_edge_cases.py

test-esci-e2e:
	$(PYTHON) tests/esci/test_e2e_mini.py

test-esci-population-e2e:
	$(PYTHON) tests/esci/test_e2e_population.py

test-esci-microbenchmark:
	$(PYTHON) tests/esci/test_microbenchmark.py

test-esci-microbenchmark-k:
	$(PYTHON) tests/esci/test_microbenchmark_k.py

# population tests (both datasets)
test-population-all: test-esci-population-e2e test-esci-microbenchmark test-esci-microbenchmark-k

# ── Utilities ─────────────────────────────────────────────────────────────────

count-datasets:
	@ls -1d datasets/synthetic/data/set*_* 2>/dev/null | wc -l

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean-cache:
	find datasets/synthetic/data -name "ground_truth_cache" -type d -exec rm -rf {} + 2>/dev/null || true

clobber: clobber-synthetic clobber-esci
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clobber-synthetic: clean-cache
	rm -rf datasets/synthetic/data/*

clobber-esci:
	find datasets/esci -maxdepth 1 -type d -name "esci_c*" -exec rm -rf {} +

# -- Q1 harness and workloads (from main) -------------------------------------
# On the instance: make q1-setup, make q1-smoke, make q1-generate DATASET=esci, then
# make q1-run-aker-b1 (and the other q1-run-* targets), make q1-collect.

PY = PYTHONPATH=. $(PYTHON)
# -- Q1 (workloads and harness) ---------------------------------------------
# On the instance: make q1-setup, make q1-smoke, make q1-generate DATASET=esci, then
# make q1-run-aker-b1 (and the other q1-run-* targets), make q1-collect.

Q1_DATA ?= /data/q1
Q1_TOOLS ?= /opt/q1
Q1_PGDATA ?= $(Q1_DATA)/pgdata
Q1_PG_PORT ?= 54320
Q1_PG_DSN ?= host=127.0.0.1 port=$(Q1_PG_PORT) user=postgres
Q1_PG_WORKERS ?= 8
Q1_DEVICE ?= auto
Q1_DRIVERS ?= build/replay
Q1_ENCODE_PY ?= $(if $(wildcard .venv-encode/bin/python),.venv-encode/bin/python,$(PYTHON))
Q1_RESULTS ?= $(Q1_DATA)/results/$(shell git rev-parse --short HEAD 2>/dev/null)
Q1_LIST_SIZE ?= 200
Q1_WORK ?= $(Q1_DATA)/work
DATASET ?= all
Q1_DATASETS ?= esci spacev sphere
Q1_BUDGETS := 1 5 10

test-q1:
	$(PYTHON) -m pytest tests/workloads tests/harness -q

build-replay:
	cmake -S harness/cpp -B build/replay -DBOOST_ROOT=$$(brew --prefix boost) -DFAISS_ROOT=$$(brew --prefix faiss) -DLIBOMP_ROOT=$$(brew --prefix libomp)
	cmake --build build/replay -j 4

# Every q1 target refuses to run from a dirty git tree (Q1_ALLOW_DIRTY=1 for local tests only).
q1-guard:
	@if [ -z "$(Q1_ALLOW_DIRTY)" ] && [ -n "$$(git status --porcelain)" ]; then \
		echo "q1: git tree is dirty; commit or stash first"; exit 1; fi

q1-setup:
	Q1_DATA=$(Q1_DATA) Q1_TOOLS=$(Q1_TOOLS) Q1_PGDATA=$(Q1_PGDATA) Q1_PG_PORT=$(Q1_PG_PORT) bash harness/deploy/setup_instance.sh

q1-pg-start:
	Q1_DATA=$(Q1_DATA) Q1_PGDATA=$(Q1_PGDATA) Q1_PG_PORT=$(Q1_PG_PORT) bash harness/deploy/pg.sh start

q1-pg-stop:
	Q1_DATA=$(Q1_DATA) Q1_PGDATA=$(Q1_PGDATA) Q1_PG_PORT=$(Q1_PG_PORT) bash harness/deploy/pg.sh stop

q1-preflight: q1-guard
	$(PY) -m harness preflight --data-dir $(Q1_DATA)

# All workload generation for one dataset (or all four): fetch, embed, float32 files, universe and
# streams, ground truth, PostgreSQL load, index, ef_search tuning, answer tables, backend recall.
q1-generate: q1-guard
	$(PY) -m harness generate --dataset $(DATASET) --data-dir $(Q1_DATA) --dsn "$(Q1_PG_DSN)" \
		--device $(Q1_DEVICE) --encode-python $(Q1_ENCODE_PY) --pg-workers $(Q1_PG_WORKERS) \
		--list-size $(Q1_LIST_SIZE) --driver-dir $(Q1_DRIVERS) \
		$(if $(GT_QUERY_CHUNK),--gt-query-chunk $(GT_QUERY_CHUNK),) \
		$(if $(GT_BASE_CHUNK),--gt-base-chunk $(GT_BASE_CHUNK),)

# One stage, e.g. make q1-stage DATASET=esci STAGE=groundtruth [FORCE=1 to redo it and later stages]
q1-stage: q1-guard
	$(PY) -m harness generate --dataset $(DATASET) --data-dir $(Q1_DATA) --dsn "$(Q1_PG_DSN)" \
		--device $(Q1_DEVICE) --encode-python $(Q1_ENCODE_PY) --pg-workers $(Q1_PG_WORKERS) \
		--list-size $(Q1_LIST_SIZE) --driver-dir $(Q1_DRIVERS) \
		--stages $(STAGE) $(if $(FORCE),--force $(STAGE),)

# One system config over every dataset and stream:
#   make q1-run SYSTEM=aker BUDGET=1
#   make q1-run SYSTEM=proximity SETTING=p0.1 BUDGET=5
q1-run: q1-guard
	@test -n "$(SYSTEM)" || { echo "set SYSTEM=aker|proximity"; exit 1; }
	@test -n "$(BUDGET)" || { echo "set BUDGET=1|5|10"; exit 1; }
	$(PY) -m harness run --datasets $(or $(DATASETS),$(Q1_DATASETS)) --systems $(SYSTEM) \
		$(if $(or $(SETTING),$(filter aker,$(SYSTEM))),--settings $(or $(SETTING),paper),) --budgets $(BUDGET) \
		--data-dir $(Q1_DATA) --out-dir $(Q1_RESULTS) --work-dir $(Q1_WORK) --driver-dir $(Q1_DRIVERS)

# Shortcuts: q1-run-aker-b{1,5,10}, q1-run-proximity-p0.1-b{1,5,10}, q1-run-proximity-p1-b{1,5,10}
Q1_RUN_TARGETS := $(foreach b,$(Q1_BUDGETS),q1-run-aker-b$(b) q1-run-proximity-p0.1-b$(b) q1-run-proximity-p1-b$(b))
.PHONY: $(Q1_RUN_TARGETS)
$(foreach b,$(Q1_BUDGETS),$(eval q1-run-aker-b$(b): ; $$(MAKE) q1-run SYSTEM=aker SETTING=paper BUDGET=$(b)))
$(foreach b,$(Q1_BUDGETS),$(eval q1-run-proximity-p0.1-b$(b): ; $$(MAKE) q1-run SYSTEM=proximity SETTING=p0.1 BUDGET=$(b)))
$(foreach b,$(Q1_BUDGETS),$(eval q1-run-proximity-p1-b$(b): ; $$(MAKE) q1-run SYSTEM=proximity SETTING=p1 BUDGET=$(b)))

# Sequential by default; make -j 9 q1-run-all runs the nine configs in parallel (each uses one core
# and shares the page cache; PostgreSQL is not needed and can be stopped with make q1-pg-stop).
q1-run-all: q1-guard $(Q1_RUN_TARGETS)

q1-collect:
	$(PY) -m harness collect --out-dir $(Q1_RESULTS) --csv $(Q1_RESULTS)/summary.csv

# Aker (paper settings) on Aker's released SPACEV stream; report only. Needs make q1-generate DATASET=spacev.
q1-fidelity: q1-guard
	$(PY) -m harness fidelity --data-dir $(Q1_DATA) --dsn "$(Q1_PG_DSN)" --driver-dir $(Q1_DRIVERS) \
		--out-dir $(Q1_DATA)/fidelity/results --work-dir $(Q1_DATA)/fidelity/work --device $(Q1_DEVICE) \
		--ef-search $$($(PYTHON) -c "import json; print(json.load(open('$(Q1_DATA)/spacev/tune.json'))['ef_search'])")

# The whole PostgreSQL cluster is snapshotted (a tar of the stopped data directory, index included).
q1-snapshot:
	Q1_DATA=$(Q1_DATA) Q1_PGDATA=$(Q1_PGDATA) Q1_PG_PORT=$(Q1_PG_PORT) bash harness/deploy/pg.sh snapshot $(or $(SNAPSHOT),$(shell date +%Y%m%d-%H%M))

q1-restore:
	@test -n "$(SNAPSHOT)" || { echo "set SNAPSHOT=<name>"; exit 1; }
	Q1_DATA=$(Q1_DATA) Q1_PGDATA=$(Q1_PGDATA) Q1_PG_PORT=$(Q1_PG_PORT) bash harness/deploy/pg.sh restore $(SNAPSHOT)

# Run any target detached from the SSH session: make q1-bg TARGET="q1-generate DATASET=esci"
# Own session (setsid when present), no terminal input, jobserver flags and pipes dropped (so make -j q1-bg
# returns at once); one log and pid file per run.
q1-bg:
	@test -n "$(TARGET)" || { echo 'set TARGET="<make target and variables>"'; exit 1; }
	@mkdir -p $(Q1_DATA)/logs
	@name="$$(echo "$(TARGET)" | tr ' =/' '___')-$$(date +%Y%m%d-%H%M%S)"; \
		log="$(Q1_DATA)/logs/$$name.log"; \
		if command -v setsid >/dev/null 2>&1; then launcher="setsid nohup"; else launcher="nohup"; fi; \
		env -u MAKEFLAGS -u MFLAGS -u MAKELEVEL $$launcher $(MAKE) $(TARGET) > "$$log" 2>&1 < /dev/null 3>&- 4>&- 5>&- 6>&- & \
		echo $$! > "$(Q1_DATA)/logs/$$name.pid"; cp "$(Q1_DATA)/logs/$$name.pid" "$(Q1_DATA)/logs/last.pid"; \
		echo "started pid $$(cat $(Q1_DATA)/logs/$$name.pid); follow with: tail -f $$log; progress: make q1-status"

q1-status:
	@$(PY) -m harness status --data-dir $(Q1_DATA)

# Smoke test of the whole stack on a tiny synthetic dataset (not part of Q1 results).
q1-smoke: q1-guard
	$(MAKE) q1-generate DATASET=tiny Q1_PG_WORKERS=2
	$(MAKE) q1-run SYSTEM=aker SETTING=paper BUDGET=1 DATASETS=tiny Q1_RESULTS=$(Q1_DATA)/smoke_results
	$(MAKE) q1-run SYSTEM=proximity SETTING=p1 BUDGET=5 DATASETS=tiny Q1_RESULTS=$(Q1_DATA)/smoke_results
	$(PY) -m harness collect --out-dir $(Q1_DATA)/smoke_results
	$(PYTHON) -m pytest tests/workloads/q1/test_core.py -q -k "torch or resident"
	@if [ -z "$(Q1_ALLOW_DIRTY)" ] && [ -n "$$(git status --porcelain)" ]; then \
		echo "q1-smoke: git status is not empty after the smoke test:"; git status --short; exit 1; fi


.PHONY: test-q1 build-replay q1-bg q1-status q1-setup q1-guard q1-preflight q1-generate q1-stage q1-run q1-run-all q1-collect q1-fidelity q1-snapshot q1-restore q1-smoke q1-pg-start q1-pg-stop

# Only the q1 targets need bash and pipefail; the jd targets keep make's default shell.
q1-%: SHELL := /bin/bash
q1-%: .SHELLFLAGS := -o pipefail -c

# ---- Experiment matrix: prepare, shard, run, reduce (recal/matrix.py; no cell logic here) ----
MATRIX ?= experiments/matrix.json
JOBS ?= jobs
SHARD_BY ?= workload
RECAL_PY ?= .venv-core/bin/python
MATRIX_FILTERS = $(if $(SYSTEM),--system $(SYSTEM),) $(if $(WORKLOAD),--workload $(WORKLOAD),)

.PHONY: prepare micro micro-inputs shards run q1 q2 q3 q4 reduce status

# make prepare DATASET=<spacev|triviaqa|medrag|tripclick|all> [STAGES=fetch,prep,working_set]: build the shared
# artifacts and push them to the store (STAGES runs only those stages)
prepare:
	@test -n "$(DATASET)" || { echo "set DATASET=<spacev|triviaqa|medrag|tripclick|all>"; exit 1; }
	$(RECAL_PY) -m scripts.prepare --config $(MATRIX) --dataset $(DATASET) $(if $(STAGES),--stages $(STAGES),)

# make micro DATASET=<spacev|triviaqa|medrag|laion> [EXPERIMENTS=E1,E3,...] [TIER=quick] [WORKERS=16] [MICRO_FLAGS=...]:
# micro-experiments (default E1 E3 E4 E7 E8 E9; also intruder_check E10 E11 E12 E13 E14), units pushed to micro/ as they
# finish. TIER=quick runs the middle budget on a sample of 2,000 held-out requests (intruder_check E1 E3 E4 E10 E11 E13).
# make micro-inputs ACTION=<push|pull|verify> DATASET=<d>: the prepared inputs in the store (a fresh instance pulls them).
MICRO_EXPERIMENTS = E1,E3,E4,E7,E8,E9
TIER ?= full
WORKERS ?= 1
micro:
	@test -n "$(DATASET)" || { echo "set DATASET=<spacev|triviaqa|medrag|laion>"; exit 1; }
	PYTHONPATH=. $(RECAL_PY) -m scripts.micro.run --dataset $(DATASET) --experiments $(or $(EXPERIMENTS),$(MICRO_EXPERIMENTS)) --tier $(TIER) --workers $(WORKERS) $(MICRO_FLAGS)

micro-inputs:
	@test -n "$(DATASET)" -a -n "$(ACTION)" || { echo "set DATASET=<d> ACTION=<push|pull|verify>"; exit 1; }
	PYTHONPATH=. $(RECAL_PY) -m scripts.micro.inputs $(ACTION) --dataset $(DATASET)

# Shard job files in $(JOBS)/shards, one per workload (SHARD_BY=question|system)
shards:
	$(RECAL_PY) -m scripts.gen_matrix --config $(MATRIX) --shard-by $(SHARD_BY) --out $(JOBS)

# make run SHARD=<name>: pull the shard artifacts, run its jobs, push the results
run:
	@test -n "$(SHARD)" || { echo "set SHARD=<name> (see $(JOBS)/shards)"; exit 1; }
	$(RECAL_PY) -m scripts.run_shard $(JOBS)/shards/$(SHARD).json

# make q1 [SYSTEM=<name or prefix>] [WORKLOAD=<name>]: the jobs of one question, optionally filtered
q1 q2 q3 q4:
	$(RECAL_PY) -m scripts.gen_matrix --config $(MATRIX) --shard-by workload --question $@ $(MATRIX_FILTERS) --out $(JOBS)/$@
	@for f in $(JOBS)/$@/shards/*.json; do case "$$f" in *.artifacts.json) ;; *) $(RECAL_PY) -m scripts.run_shard $$f || exit 1;; esac; done

reduce:
	$(RECAL_PY) -m scripts.reduce --config $(MATRIX) --jobs $(JOBS) --out figures_data

status:
	$(RECAL_PY) -m scripts.status --jobs $(JOBS)
