#!/bin/sh
# Run a job list to completion: mx_complete.sh <list> <flag> [slots]
# Runs every line through mx_run_job.sh (xargs, <slots> at a time), then checks the OUTPUTS: a line
# is complete when its JSON exists, it is in /tmp/mx/skip.txt, or its log has a traceback (a real
# failure, reported). Lines still missing are run again (up to 5 rounds). Only then is <flag>
# touched, with <flag>.missing listing anything still absent (11-13: flags used to be set when a
# list had been worked through, even if its jobs had died). A JSON counts only if its config_hash
# matches the line (stale ones are rerun; MX_TRUST_EXISTING=1 restores the existence check).
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
LIST=$1; FLAG=$2; SLOTS=${3:-24}
W=simulations/approximate_hits/mx_run_job.sh
PYBIN=${MX_PY:-$PWD/.venv-core/bin/python}
[ -x "$PYBIN" ] || PYBIN=python3
CHECK_FLAGS=""
[ -n "$MX_STRICT_HASH" ] && CHECK_FLAGS="$CHECK_FLAGS --strict-hash"
[ -n "$MX_TRUST_EXISTING" ] && CHECK_FLAGS="$CHECK_FLAGS --trust-existing"
fresh() {   # fresh <json> <line>: 0 when the result exists and matches the line's config_hash
  [ -f "$1" ] || return 1
  PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} "$PYBIN" -m recal.measure.jobs_util check "$1" "$2" $CHECK_FLAGS 2>/dev/null; r=$?
  case $r in 0) return 0;; 2) return 1;; esac
  return 0   # checker could not run: existence only
}
todo=${RECAL_QUEUE_DIR:-/tmp/mx}/$(basename $FLAG).todo
cp "$LIST" $todo
for round in 1 2 3 4 5; do
  xargs -P "$SLOTS" -I{} sh $W 0 {} < $todo || echo "$(date +%T) round $round: some jobs failed (xargs rc $?)"
  : > $todo.next
  while IFS= read -r l; do
    j=$(printf '%s' "$l" | grep -o "\[ -f [^ ]*json" | cut -d' ' -f3)
    g=$(printf '%s' "$l" | grep -oE "> [^ ]*logs/[^ ]*\.txt" | cut -c3-)
    [ -n "$j" ] && fresh "$j" "$l" && continue
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
