#!/bin/sh
# Memory-guarded runner for one job line of the matrix drivers (run_matrix*.sh, probes).
# Usage: mx_run_job.sh <ignored> '<job command line>'
# The job line has the form "mkdir -p ... && [ -f <out>.json ] || <command> > <log> 2>&1".
#  - skips the job if its JSON exists or is listed in /tmp/mx/skip.txt (dropped from the plan),
#    or waits while an identical job (same JSON
#    target) is already running (e.g. an orphan of an earlier driver);
#  - GLOBAL admission (v3, 11-10): a job starts only if (memory committed by running jobs +
#    its own peak estimate) <= CAP (320 GB of 371) and MemAvailable >= its estimate + 30 GB, checked
#    under a lock (flock) so two jobs cannot admit on the same reading; nothing starts while
#    /tmp/mx/HOLD exists (a prep step is allocating). Commitment: see committed(); peak
#    estimates (private GB, measured 11-10): ours at a 10% budget 18 (warm-up build), LRU at 10%
#    10, other simulator jobs at 10% 12, at 1% 6, ESCI 5, real Aker 3 (TriviaQA 6), prep 90;
#    11-13: 3% budget 12 (ours) / 8; TriviaQA simulator jobs x2 (769-d vectors; 3% runs were killed).
#    Per-pool caps (xargs -P) only bound how many jobs a pool keeps waiting;
#  - if it ends without its JSON (e.g. killed for memory, which leaves no traceback),
#    waits and retries, at most 2 retries; a job that wrote a traceback is not retried.
# (The first argument was the old per-job free-memory threshold; kept for compatibility.)
LINE=$2
CAP=320
HEADROOM=30     # MemAvailable must also exceed the job's estimate by this much (jobs grow after admission)
JSON=$(printf '%s' "$LINE" | grep -o "\[ -f [^ ]*json" | cut -d' ' -f3)
LOG=$(printf '%s' "$LINE" | grep -oE "> [^ ]*logs/[^ ]*\.txt" | cut -c3-)
LOCK=/tmp/mx/admit.lock

est_of() {   # PEAK private memory estimate (GB) of a job class, from its command line
  printf '%s' "$1" | awk '
    /eval_aker_real.py/ { print (/triviaqa/ ? 6 : 3); exit }
    /--budget 0\.1 / && /--method (static|greedy)/ { print (/triviaqa/ ? 36 : 18); exit }
    /--budget 0\.1 / && /--method lru/ { print (/triviaqa/ ? 20 : 10); exit }
    /--budget 0\.1 / { print (/triviaqa/ ? 24 : 12); exit }
    /--budget 0\.03 / && /--method (static|greedy)/ { print (/triviaqa/ ? 24 : 12); exit }
    /--budget 0\.03 / { print (/triviaqa/ ? 16 : 8); exit }
    /triviaqa/ { print 12; exit }
    /esci1m/ { print 5; exit }
    { print 6 }'
}
committed() {   # GB committed by running jobs and prep steps
  # a job in its first 30 min counts at max(class peak estimate, private memory); later (past the
  # warm-up build, where the peak occurs) at 1.2 x its private memory; a prep step at
  # max(90, private). Private = Private_Clean + Private_Dirty (/proc/<pid>/smaps_rollup): RSS would
  # count the shared memory-mapped database once per process.
  ps -eo pid=,etimes=,args= | awk '
    /^ *[0-9]+ +[0-9]+ [^ ]*python[^ ]* -u [^ ]*(online_sim|eval_aker_real)\.py/ {
      line = $0
      if (line ~ /eval_aker_real.py/) e = (line ~ /triviaqa/ ? 6 : 3)
      else if (line ~ /--budget 0\.1 / && line ~ /--method (static|greedy)/) e = 18
      else if (line ~ /--budget 0\.1 / && line ~ /--method lru/) e = 10
      else if (line ~ /--budget 0\.1 /) e = 12
      else if (line ~ /--budget 0\.03 / && line ~ /--method (static|greedy)/) e = 12
      else if (line ~ /--budget 0\.03 /) e = 8
      else if (line ~ /esci1m/) e = 5
      else e = 6
      if (line ~ /triviaqa/ && line !~ /eval_aker_real.py/) e = 2 * e     # 769-d vectors (11-13)
      print $1, $2, e }
    /^ *[0-9]+ +[0-9]+ [^ ]*python[^ ]* -u [^ ]*online_prepare(_data)?\.py/ { print $1, -1, 90 }' |
  while read pid age e; do
    priv=$(awk '/^Private_(Clean|Dirty):/ {s += $2} END {printf "%d", s / 1048576 + 0.5}' /proc/$pid/smaps_rollup 2>/dev/null)
    [ -n "$priv" ] || continue
    if [ "$age" -ge 1800 ]; then echo $(( priv + priv / 5 + 1 ))
    elif [ "$priv" -gt "$e" ]; then echo "$priv"; else echo "$e"; fi
  done | awk '{s += $1} END {printf "%d\n", s}'
}
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
same_running() {   # another process writing the same JSON target is alive
  d=$(dirname "$JSON"); t=$(basename "$JSON" .json); tag=_${t#*_}
  ps -eo args | grep -v "sh -c" | grep -v mx_run_job | grep -q -- "--tag $tag --out $d\( \|$\)" && return 0
  ps -eo args | grep -v "sh -c" | grep -v mx_run_job | grep -q -- "--out $JSON\( \|$\)"
}

[ -n "$JSON" ] && [ -f "$JSON" ] && exit 0
# jobs dropped from the plan (e.g. the detector factorial on TriviaQA, see doc 9j)
[ -n "$JSON" ] && [ -f /tmp/mx/skip.txt ] && grep -qxF "$JSON" /tmp/mx/skip.txt && exit 0
while [ -n "$JSON" ] && same_running; do sleep 60; done
[ -n "$JSON" ] && [ -f "$JSON" ] && exit 0
NEED=$(est_of "$LINE")
try=0
while :; do
  # admission: under the lock, check the global budget, start the job, wait until its
  # process is visible (so the next admission counts it), release the lock
  admitted=0
  # re-check the plan on every attempt (a job can be dropped while it waits or between retries)
  [ -n "$JSON" ] && [ -f /tmp/mx/skip.txt ] && grep -qxF "$JSON" /tmp/mx/skip.txt && exit 0
  exec 9> $LOCK
  if flock -w 30 9; then
    if [ ! -f /tmp/mx/HOLD ] && [ $(( $(committed) + NEED )) -le $CAP ] && [ "$(free_gb)" -ge $(( NEED + HEADROOM )) ]; then
      sh -c "$LINE" &
      job=$!
      for i in 1 2 3 4 5 6 7 8 9 10; do
        pgrep -P $job > /dev/null 2>&1 && break
        sleep 1
      done
      sleep 2
      admitted=1
    fi
    flock -u 9
  fi
  exec 9>&-
  if [ $admitted -eq 0 ]; then
    sleep $((20 + $(od -An -N1 -tu1 /dev/urandom) % 40))
    continue
  fi
  wait $job
  [ -z "$JSON" ] || [ -f "$JSON" ] && exit 0
  if [ -n "$LOG" ] && grep -q "Traceback" "$LOG" 2>/dev/null; then
    echo "$(date +%T) FAILED (traceback) $JSON" >> /tmp/mx/failures.log; exit 1
  fi
  try=$((try + 1))
  echo "$(date +%T) died without output (try $try) $JSON" >> /tmp/mx/failures.log
  [ $try -gt 2 ] && { echo "$(date +%T) GAVE UP $JSON" >> /tmp/mx/failures.log; exit 1; }
  sleep 120
done
