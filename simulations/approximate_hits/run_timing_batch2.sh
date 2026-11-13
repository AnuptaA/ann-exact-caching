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
# quiet machine by suspension (v6, 11-13): no queued job is admitted (HOLD), and every other python
# process of ours (registered or not) is suspended (SIGSTOP) for the batch and resumed at the end;
# one started meanwhile is suspended within a minute. The timing runs register as priority
# "timing" (never suspended or killed). The quiet-machine guard of each timing run (--wait-quiet)
# then decides, by measurement, when the machine is quiet.
REG=$M/registry; mkdir -p $REG
: > $M/$NAME.stopped
suspend_all() {   # every python process of ours except the timing runs (registry priority timing)
  for p in $(ps -u "$(id -u)" -o pid=,comm= | awk '$2 ~ /^python/ {print $1}'); do
    [ -f $REG/$p ] && [ "$(awk '{print $2}' $REG/$p)" = timing ] && continue
    grep -qx "$p" $M/$NAME.stopped || { kill -STOP $p 2>/dev/null && echo $p >> $M/$NAME.stopped; }
  done
}
( while [ -d $M/prep.lock ] && [ ! -f $M/$NAME.DONE ]; do touch $M/HOLD; suspend_all; sleep 60; done ) &
keeper=$!
suspend_all
echo "$(date +%T) prep lock taken, HOLD set; suspended $(grep -c . $M/$NAME.stopped) jobs"
while IFS= read -r line; do
  echo "$(date +%T) start: $(printf '%s' "$line" | grep -o -- '--out [^ ]*' | cut -d' ' -f2)"
  sh -c "$line" &
  job=$!
  # register the timing run before the keeper's next pass (it suspends any unregistered python)
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    for c in $(pgrep -P $job) $(for g in $(pgrep -P $job); do pgrep -P $g; done); do
      case "$(cat /proc/$c/comm 2>/dev/null)" in python*) echo "timing timing 0" > $REG/$c; kill -CONT $c ;; esac
    done
    sleep 0.5
  done
  wait $job
done < $M/$NAME.txt

touch $M/$NAME.DONE
kill $keeper 2>/dev/null
for p in $(cat $M/$NAME.stopped); do kill -CONT $p 2>/dev/null; done
echo "$(date +%T) resumed $(grep -c . $M/$NAME.stopped) jobs"
rm -f $M/HOLD
rmdir $M/prep.lock 2>/dev/null
echo "$(date +%T) done"
