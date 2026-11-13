#!/bin/sh
# Memory-guarded runner for one job line of the matrix drivers (run_matrix*.sh, probes, hx / hr).
# Usage: mx_run_job.sh <ignored> '<job command line>'
# The job line has the form "mkdir -p ... && [ -f <out>.json ] || <command> > <log> 2>&1".
#  - skips the job if its JSON exists or is listed in /tmp/mx/skip.txt (dropped from the plan),
#    or waits while an identical job (same JSON target) is already running;
#  - GLOBAL admission under a lock (flock): a job starts only if (memory committed by running
#    jobs + its own peak estimate) <= CAP and MemAvailable >= its estimate + HEADROOM; nothing
#    starts while /tmp/mx/HOLD exists;
#  - v4 (11-13, self-adjusting): the peak estimate of a job class (prep, budget, method) is LEARNED:
#    every run's peak private memory is recorded in /tmp/mx/peaks.tsv (sampled every 20 s); the
#    estimate is max(table default, 1.3 x the largest finished peak, 1.6 x the largest peak of a
#    run that was killed, which was still growing). The table below is only the prior for classes
#    never seen;
#  - a run that ends without its JSON and without a traceback (killed for memory, by the watchdog
#    or the kernel) is retried, with back-off 2, 4, 8, 16, 32 min (at most 20 attempts): memory
#    pressure costs time, not results. A run that wrote a traceback is not retried.
LINE=$2
CAP=320
HEADROOM=30
PEAKS=/tmp/mx/peaks.tsv
JSON=$(printf '%s' "$LINE" | grep -o "\[ -f [^ ]*json" | cut -d' ' -f3)
LOG=$(printf '%s' "$LINE" | grep -oE "> [^ ]*logs/[^ ]*\.txt" | cut -c3-)
LOCK=/tmp/mx/admit.lock

jkey() {   # job class: prep folder, budget, method (from a command line)
  printf '%s' "$1" | awk '{
    p = "?"; b = "?"; m = "?"
    if (match($0, /--prep [^ ]*/)) { p = substr($0, RSTART + 7, RLENGTH - 7); sub(/.*\//, "", p) }
    if (match($0, /--budget [0-9.]+/)) b = substr($0, RSTART + 9, RLENGTH - 9)
    if ($0 ~ /eval_aker_real.py/) m = "aker_real"
    else if (match($0, /--method [a-z]+/)) m = substr($0, RSTART + 9, RLENGTH - 9)
    if ($0 ~ /--save-state/) m = m "_state"
    print p "|" b "|" m }'
}
prior_of() {   # prior peak estimate (GB) for a class never seen
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
est_of() {   # peak estimate (GB): learned from recorded peaks of the class, the prior otherwise
  k=$(jkey "$1"); pr=$(prior_of "$1")
  awk -F'\t' -v k="$k" -v pr="$pr" '
    $1 == k && $3 == "ok" && $2 > ok { ok = $2 }
    $1 == k && $3 == "killed" && $2 > kd { kd = $2 }
    END { e = pr; if (1.3 * ok > e) e = 1.3 * ok; if (1.6 * kd > e) e = 1.6 * kd
          printf "%d\n", e + 0.999 }' "$PEAKS" 2>/dev/null || echo "$pr"
}
priv_of() {   # private memory (GB, rounded up) of a pid
  awk '/^Private_(Clean|Dirty):/ {s += $2} END {printf "%d", s / 1048576 + 0.999}' /proc/$1/smaps_rollup 2>/dev/null
}
committed() {   # GB committed by running jobs and prep steps
  # a job in its first 30 min counts at max(class estimate, private memory); later at 1.2 x its
  # private memory; a prep step at max(90, private). Private = Private_Clean + Private_Dirty.
  # also processes outside the queue (index builds / tuning, latency probes): at max(60, private)
  # while young, as builds briefly reach their memory limit (11-13: two builds and a tuning run,
  # not counted, drove MemAvailable to 0)
  ps -eo pid=,etimes=,args= | grep -E "^ *[0-9]+ +[0-9]+ [^ ]*python[^ ]* -u [^ ]*(online_sim|eval_aker_real|online_prepare|online_prepare_data|eval_real_backend|index_study|latency_probe|eval_timing_online)\.py" |
  while read pid age args; do
    priv=$(priv_of $pid); [ -n "$priv" ] || continue
    case "$args" in
      *online_prepare*) e=90 ;;
      *eval_real_backend*|*index_study*) e=60 ;;
      *latency_probe*|*eval_timing_online*) e=12 ;;
      *) e=$(est_of "$args") ;;
    esac
    if [ "$age" -ge 1800 ] && [ "$e" -ne 90 ] && [ "$e" -ne 60 ]; then echo $(( priv + priv / 5 + 1 ))
    elif [ "$priv" -gt "$e" ]; then echo "$priv"; else echo "$e"; fi
  done | awk '{s += $1} END {printf "%d\n", s}'
}
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
same_running() {   # another process writing the same JSON target is alive
  d=$(dirname "$JSON"); t=$(basename "$JSON" .json); tag=_${t#*_}
  ps -eo args | grep -v "sh -c" | grep -v mx_run_job | grep -q -- "--tag $tag --out $d\( \|$\)" && return 0
  ps -eo args | grep -v "sh -c" | grep -v mx_run_job | grep -q -- "--out $JSON\( \|$\)"
}
skipped() { [ -n "$JSON" ] && [ -f /tmp/mx/skip.txt ] && grep -qxF "$JSON" /tmp/mx/skip.txt; }

[ -n "$JSON" ] && [ -f "$JSON" ] && exit 0
skipped && exit 0
while [ -n "$JSON" ] && same_running; do sleep 60; done
[ -n "$JSON" ] && [ -f "$JSON" ] && exit 0
KEY=$(jkey "$LINE")
try=0
while :; do
  skipped && exit 0
  NEED=$(est_of "$LINE")
  admitted=0
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
  # watch the run: its peak private memory (the python process under sh -c)
  peak=0
  while kill -0 $job 2>/dev/null; do
    for c in $(pgrep -P $job); do
      p=$(priv_of $c); [ -n "$p" ] && [ "$p" -gt "$peak" ] && peak=$p
    done
    sleep 20
  done
  wait $job 2>/dev/null
  if [ -z "$JSON" ] || [ -f "$JSON" ]; then
    [ "$peak" -gt 0 ] && printf '%s\t%s\tok\n' "$KEY" "$peak" >> $PEAKS
    exit 0
  fi
  if [ -n "$LOG" ] && grep -q "Traceback" "$LOG" 2>/dev/null; then
    echo "$(date +%T) FAILED (traceback) $JSON" >> /tmp/mx/failures.log; exit 1
  fi
  [ "$peak" -gt 0 ] && printf '%s\t%s\tkilled\n' "$KEY" "$peak" >> $PEAKS
  try=$((try + 1))
  echo "$(date +%T) died without output (try $try, peak ${peak} GB, next estimate $(est_of "$LINE") GB) $JSON" >> /tmp/mx/failures.log
  [ $try -ge 20 ] && { echo "$(date +%T) GAVE UP after $try attempts $JSON" >> /tmp/mx/failures.log; exit 1; }
  b=$try; [ $b -gt 5 ] && b=5
  sleep $(( 60 * (1 << b) ))
done
