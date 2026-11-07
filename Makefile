VENV ?= $(if $(wildcard .venv/bin/python),.venv,venv)
PYTHON = $(VENV)/bin/python
PY = PYTHONPATH=. $(PYTHON)
AH = simulations/approximate_hits
CLI = $(PY) -m simulations.approximate_hits

# Run directory used by the evaluation targets (written by populate-*).
RUN ?= $(AH)/results/esci_score20k
STREAM ?= shift
METHOD ?= triggered

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

.PHONY: all help
.PHONY: populate-esci populate-sift eval-core eval-final eval-shell
.PHONY: online-prepare online-run online-summary figures
.PHONY: test-q1 build-replay
.PHONY: q1-bg q1-setup q1-guard q1-preflight q1-generate q1-stage q1-run q1-run-all q1-collect q1-fidelity q1-snapshot q1-restore q1-smoke q1-pg-start q1-pg-stop
.PHONY: test test-cache test-approximate

all: help

help:
	@echo "Populate:  populate-esci | populate-sift      (writes results.json into RUN)"
	@echo "Evaluate:  eval-core | eval-final | eval-shell   (RUN=<run dir>)"
	@echo "Online:    online-prepare | online-run METHOD=<static|periodic|triggered|lru|aker|qvcache> | online-summary"
	@echo "Figures:   figures"
	@echo "Q1:        q1-setup | q1-smoke | q1-generate DATASET=<d> | q1-run-<aker|proximity-p0.1|proximity-p1>-b<1|5|10> | q1-collect | q1-fidelity (see harness/README.md)"
	@echo "Tests:     test | test-cache | test-approximate | test-q1"
	@echo "Retired exact-caching pipelines live in stale/ (see stale/README.md)."



# -- Populate and evaluate --------------------------------------------------

populate-esci:
	$(CLI) populate esci --out $(RUN)

populate-sift:
	$(CLI) populate sift --out $(RUN)

eval-core:
	$(CLI) eval core $(RUN)

eval-final:
	$(CLI) eval final $(RUN)

eval-shell:
	$(CLI) eval simple $(RUN)
	$(CLI) eval shell-ablation $(RUN)

# -- Online simulation ------------------------------------------------------

online-prepare:
	$(CLI) online prepare

online-run:
	$(CLI) online run --method $(METHOD) --stream $(STREAM)

online-summary:
	$(CLI) online summary $(AH)/results/online/esci1m/sim_$(STREAM)_b0.1

# -- Figures ----------------------------------------------------------------

figures:
	$(CLI) report figures

# -- Tests ------------------------------------------------------------------

test:
	$(PYTHON) -m pytest tests/ -q


test-cache:
	$(PYTHON) -m pytest tests/cache/ -q

test-approximate:
	$(PYTHON) -m pytest tests/approximate_hits/ -q



# -- Q1 (workloads and harness) ---------------------------------------------
# On the instance: make q1-setup, make q1-smoke, make q1-generate DATASET=esci, then
# make q1-run-aker-b1 (and the other q1-run-* targets), make q1-collect.

Q1_DATA ?= /data/q1
Q1_PGDATA ?= $(Q1_DATA)/pgdata
Q1_PG_PORT ?= 5432
Q1_PG_DSN ?= host=127.0.0.1 port=$(Q1_PG_PORT) user=postgres
Q1_PG_WORKERS ?= 8
Q1_DEVICE ?= auto
Q1_DRIVERS ?= build/replay
Q1_ENCODE_PY ?= $(if $(wildcard .venv-encode/bin/python),.venv-encode/bin/python,$(PYTHON))
Q1_RESULTS ?= $(Q1_DATA)/results/$(shell git rev-parse --short HEAD 2>/dev/null)
Q1_LIST_SIZE ?= 200
Q1_WORK ?= $(Q1_DATA)/work
DATASET ?= all
Q1_DATASETS ?= esci spacev sift10m sphere
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
	Q1_DATA=$(Q1_DATA) Q1_PGDATA=$(Q1_PGDATA) Q1_PG_PORT=$(Q1_PG_PORT) bash harness/deploy/setup_instance.sh

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
		--list-size $(Q1_LIST_SIZE) --driver-dir $(Q1_DRIVERS) $(if $(Q1_ALLOW_SLOW),--allow-slow,) \
		$(if $(GT_QUERY_CHUNK),--gt-query-chunk $(GT_QUERY_CHUNK),) \
		$(if $(GT_BASE_CHUNK),--gt-base-chunk $(GT_BASE_CHUNK),)

# One stage, e.g. make q1-stage DATASET=esci STAGE=groundtruth [FORCE=1 to redo it and later stages]
q1-stage: q1-guard
	$(PY) -m harness generate --dataset $(DATASET) --data-dir $(Q1_DATA) --dsn "$(Q1_PG_DSN)" \
		--device $(Q1_DEVICE) --encode-python $(Q1_ENCODE_PY) --pg-workers $(Q1_PG_WORKERS) \
		--list-size $(Q1_LIST_SIZE) --driver-dir $(Q1_DRIVERS) $(if $(Q1_ALLOW_SLOW),--allow-slow,) \
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
q1-bg:
	@test -n "$(TARGET)" || { echo 'set TARGET="<make target and variables>"'; exit 1; }
	@mkdir -p $(Q1_DATA)/logs
	@log="$(Q1_DATA)/logs/$$(echo "$(TARGET)" | tr ' =/' '___')-$$(date +%Y%m%d-%H%M%S).log"; \
		nohup $(MAKE) $(TARGET) > "$$log" 2>&1 & echo $$! > "$(Q1_DATA)/logs/last.pid"; \
		echo "started pid $$(cat $(Q1_DATA)/logs/last.pid); follow with: tail -f $$log"

# Smoke test of the whole stack on a tiny synthetic dataset (not part of Q1 results).
q1-smoke: q1-guard
	$(MAKE) q1-generate DATASET=tiny Q1_DEVICE=numpy Q1_PG_WORKERS=2
	$(MAKE) q1-run SYSTEM=aker SETTING=paper BUDGET=1 DATASETS=tiny Q1_RESULTS=$(Q1_DATA)/smoke_results
	$(MAKE) q1-run SYSTEM=proximity SETTING=p1 BUDGET=5 DATASETS=tiny Q1_RESULTS=$(Q1_DATA)/smoke_results
	$(PY) -m harness collect --out-dir $(Q1_DATA)/smoke_results
	$(PYTHON) -m pytest tests/workloads/q1/test_core.py -q -k "torch or resident"

