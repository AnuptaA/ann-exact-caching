#!/bin/sh
# Run a job list to completion: mx_complete.sh <list> <flag> [slots]
# Runs every line through mx_run_job.sh (xargs, <slots> at a time), then checks the OUTPUTS: a line
# is complete when its JSON exists, it is in /tmp/mx/skip.txt, or its log has a traceback (a real
# failure, reported). Lines still missing are run again (up to 5 rounds). Only then is <flag>
# touched, with <flag>.missing listing anything still absent (11-13: flags used to be set when a
# list had been worked through, even if its jobs had died).
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
LIST=$1; FLAG=$2; SLOTS=${3:-24}
W=simulations/approximate_hits/mx_run_job.sh
todo=${RECAL_QUEUE_DIR:-/tmp/mx}/$(basename $FLAG).todo
cp "$LIST" $todo
for round in 1 2 3 4 5; do
  xargs -P "$SLOTS" -I{} sh $W 0 {} < $todo
  : > $todo.next
  while IFS= read -r l; do
    j=$(printf '%s' "$l" | grep -o "\[ -f [^ ]*json" | cut -d' ' -f3)
    g=$(printf '%s' "$l" | grep -oE "> [^ ]*logs/[^ ]*\.txt" | cut -c3-)
    [ -n "$j" ] && [ -f "$j" ] && continue
    [ -n "$j" ] && [ -f ${RECAL_QUEUE_DIR:-/tmp/mx}/skip.txt ] && grep -qxF "$j" ${RECAL_QUEUE_DIR:-/tmp/mx}/skip.txt && continue
    [ -n "$g" ] && grep -q Traceback "$g" 2>/dev/null && { echo "traceback: $j" >> $FLAG.failed; continue; }
    printf '%s\n' "$l" >> $todo.next
  done < "$LIST"
  mv $todo.next $todo
  n=$(grep -c . $todo)
  echo "$(date +%T) $(basename $LIST) round $round: $n missing"
  [ "$n" -eq 0 ] && break
done
cp $todo $FLAG.missing
touch $FLAG
