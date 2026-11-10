#!/bin/sh
# Memory-guarded runner for one job line of the matrix driver (run_matrix.sh).
# Usage: mx_run_job.sh <min_free_gb> '<job command line>'
# The job line has the form "mkdir -p ... && [ -f <out>.json ] || <command> > <log> 2>&1".
#  - skips the job if its JSON exists, or waits while an identical job (same JSON
#    target) is already running (e.g. an orphan of an earlier driver);
#  - starts it only when MemAvailable >= <min_free_gb> (re-checked after a random
#    pause, so many slots do not start at once on the same reading);
#  - if it ends without its JSON (e.g. killed by the out-of-memory killer, which
#    leaves no traceback), waits for memory and retries, at most 2 retries; a
#    job that writes a Python traceback is not retried.
MIN=$1
LINE=$2
JSON=$(printf '%s' "$LINE" | grep -o "\[ -f [^ ]*json" | cut -d' ' -f3)
LOG=$(printf '%s' "$LINE" | grep -oE "> [^ ]*logs/[^ ]*\.txt" | cut -c3-)
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
same_running() {   # another process writing the same JSON target is alive
  d=$(dirname "$JSON"); t=$(basename "$JSON" .json); tag=_${t#*_}
  ps -eo args | grep -v "sh -c" | grep -v mx_run_job | grep -q -- "--tag $tag --out $d\( \|$\)" && return 0
  ps -eo args | grep -v "sh -c" | grep -v mx_run_job | grep -q -- "--out $JSON\( \|$\)"
}
[ -n "$JSON" ] && [ -f "$JSON" ] && exit 0
while [ -n "$JSON" ] && same_running; do sleep 60; done
[ -n "$JSON" ] && [ -f "$JSON" ] && exit 0
try=0
while :; do
  while [ "$(free_gb)" -lt "$MIN" ]; do sleep $((30 + $(od -An -N1 -tu1 /dev/urandom) % 30)); done
  sleep $(( $(od -An -N1 -tu1 /dev/urandom) % 20 ))
  [ "$(free_gb)" -lt "$MIN" ] && continue
  sh -c "$LINE"
  [ -z "$JSON" ] || [ -f "$JSON" ] && exit 0
  if [ -n "$LOG" ] && grep -q "Traceback" "$LOG" 2>/dev/null; then
    echo "$(date +%T) FAILED (traceback) $JSON" >> /tmp/mx/failures.log; exit 1
  fi
  try=$((try + 1))
  echo "$(date +%T) died without output (try $try) $JSON" >> /tmp/mx/failures.log
  [ $try -gt 2 ] && { echo "$(date +%T) GAVE UP $JSON" >> /tmp/mx/failures.log; exit 1; }
  sleep 120
done
