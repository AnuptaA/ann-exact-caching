#!/bin/sh
# Generic timing batch: run_timing_batch2.sh <timing_jobs set> <name> <flags to wait for>...
# Timing batch for the timing plot (context doc 9k; v3 full-stream design, doc 9m), run detached:
#   setsid nohup sh simulations/approximate_hits/run_timing_batch.sh > /tmp/mx/timing.log 2>&1 &
# 1. waits for its inputs: the tuned-Aker runs (Aker's setting is picked from them) and the
#    warm-up states of ours / LRU (online_sim --save-state with DiskANN's answers);
# 2. generates the job lines (timing_jobs.py -> /tmp/mx/timing.txt);
# 3. quiets the machine: takes the prep lock (no index build / prep can start), keeps /tmp/mx/HOLD
#    in place (no queued job is admitted; re-touched every minute, since other scripts remove it),
#    and waits until no simulator / Aker / prep / index-build process is left;
# 4. runs the jobs one at a time, each under the quiet-machine guard (--wait-quiet 30);
# 5. releases HOLD and the prep lock.
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx
S=simulations/approximate_hits
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.

busy() {   # any running job that would disturb a timing run: python processes only (queued
  # mx_run_job.sh wrappers carry the same script names in their arguments but only wait on HOLD;
  # matching them deadlocked the first batch, 11-11)
  pgrep -f "^[^ ]*python[^ ]* -u [^ ]*(online_sim|eval_aker_real|online_prepare|online_prepare_data|eval_real_backend|reembed_triviaqa_queries)\.py" > /dev/null
}

echo "$(date +%T) waiting for the tuned-Aker runs and the warm-up states"
SET=$1; NAME=$2; shift 2
for f in "$@"; do while [ ! -f $M/$f ]; do sleep 300; done; done
# LRU runs that started before the empty-search fix (doc 9n) are redone
for f in $S/results/timing_v3/*/lru_s*.json; do
  [ -f "$f" ] && [ "$f" -ot $M/LRU_FIX.marker ] && mkdir -p $S/results/timing_v3/old_lru_before_fix && mv "$f" "${f%.json}_lat.npz" $S/results/timing_v3/old_lru_before_fix/ 2>/dev/null
done
PYTHONPATH=. .venv/bin/python $S/timing_jobs.py --set $SET > $M/$NAME.txt || exit 1
echo "$(date +%T) $(wc -l < $M/$NAME.txt) timing jobs ($SET)"

[ -n "$HAVE_LOCK" ] || until mkdir $M/prep.lock 2>/dev/null; do sleep 120; done   # HAVE_LOCK=1: restart holding it
( while [ -d $M/prep.lock ] && [ ! -f $M/$NAME.DONE ]; do touch $M/HOLD; sleep 60; done ) &
keeper=$!
echo "$(date +%T) prep lock taken, HOLD set; draining"
while busy; do sleep 120; done
echo "$(date +%T) drained; timing"

while IFS= read -r line; do
  echo "$(date +%T) start: $(printf '%s' "$line" | grep -o -- '--out [^ ]*' | cut -d' ' -f2)"
  sh -c "$line"
done < $M/$NAME.txt

touch $M/$NAME.DONE
kill $keeper 2>/dev/null
rm -f $M/HOLD
rmdir $M/prep.lock 2>/dev/null
echo "$(date +%T) done"
