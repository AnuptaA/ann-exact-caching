#!/bin/sh
# Memory-guarded runner for one job (v5, 11-13): mx_run_job.sh <ignored> '<job command line>'
# A job line is a shell command; when it contains "[ -f <out>.json ]" that JSON is its output.
# Priority: env MX_PRIORITY = plot (default; feeds the paper's plots) or adhoc (exploratory).
#  - skip: the JSON exists, or it is listed in /tmp/mx/skip.txt; waits while the same JSON target
#    is being produced by another live process;
#  - registry: the job's python processes are recorded in /tmp/mx/registry/<pid> ("key priority"),
#    used by admission, by the watchdog (adhoc jobs and unregistered heavy processes are killed
#    first) and by the timing batches (suspension); oom_score_adj: plot 300, adhoc 1000, so the
#    kernel's OOM killer also prefers ad-hoc work;
#  - admission, from the MACHINE's state, not from bookkeeping: used = MemTotal - MemAvailable
#    (everything running, ours or not) plus the expected growth of every registered job
#    (max(0, its class estimate - its private memory), times the factor F in /tmp/mx/growth_factor,
#    which the watchdog adapts: +0.25 per memory kill (up to 1), -0.1 per quiet hour with free
#    memory (down to 0.2)); a job starts only if used + growth + its
#    own estimate <= CAP and MemAvailable - growth - estimate >= HEADROOM. Nothing starts while
#    /tmp/mx/HOLD exists. Admissions are serialized by a lock;
#  - class estimates (prep, budget, method) are learned from recorded peaks (/tmp/mx/peaks.tsv):
#    max(prior, 1.2 x largest finished peak, 1.1 x largest killed peak) (a killed run's peak is a
#    lower bound on its need, not evidence of more: kills follow the machine being full; 1.6 x
#    ratcheted estimates up under pressure);
#  - retries: a JSON job that ends without its JSON and without a traceback (killed) is retried
#    with back-off 2-32 min, up to 20 attempts; a job without a JSON succeeds on exit status 0 and
#    is retried the same way on a kill signal (exit > 128).
LINE=$2
PRIO=${MX_PRIORITY:-plot}
CAP=345
HEADROOM=20
PEAKS=${RECAL_QUEUE_DIR:-/tmp/mx}/peaks.tsv
REG=${RECAL_QUEUE_DIR:-/tmp/mx}/registry
mkdir -p $REG
JSON=$(printf '%s' "$LINE" | grep -o "\[ -f [^ ]*json" | cut -d' ' -f3)
LOG=$(printf '%s' "$LINE" | grep -oE "> [^ ]*logs/[^ ]*\.txt" | cut -c3-)
LOCK=${RECAL_QUEUE_DIR:-/tmp/mx}/admit.lock

jkey() {   # job class: prep folder, budget, method / script (from a command line)
  printf '%s' "$1" | awk '{
    p = "?"; b = "-"; m = "?"
    if (match($0, /--prep [^ ]*/)) { p = substr($0, RSTART + 7, RLENGTH - 7); sub(/.*\//, "", p) }
    if (match($0, /--budget [0-9.]+/)) b = substr($0, RSTART + 9, RLENGTH - 9)
    if (match($0, /--method [a-z]+/)) m = substr($0, RSTART + 9, RLENGTH - 9)
    else if (match($0, /[a-z_]+\.py/)) m = substr($0, RSTART, RLENGTH - 3)
    if ($0 ~ /eval_aker_real.py/) m = "aker_real"
    if ($0 ~ /--save-state/) m = m "_state"
    if (match($0, /--variant [a-z0-9]+/)) m = m "_" substr($0, RSTART + 10, RLENGTH - 10)
    print p "|" b "|" m }'
}
prior_of() {   # prior peak estimate (GB) for a class never seen: modest, since an underestimate
  # costs only a retry (memory kills are retried) and the learned peaks take over after one run;
  # high only for classes known to be heavy (index builds / tuning, ours and LRU at 10%)
  printf '%s' "$1" | awk '
    /index_study.py|eval_real_backend.py/ { print 60; exit }
    /online_prepare/ { print 90; exit }
    /eval_aker_real.py/ { print (/triviaqa/ ? 6 : 3); exit }
    /--budget 0\.1 / && /--method (static|greedy|lru)/ { print (/triviaqa/ ? 36 : 24); exit }
    /--budget 0\.03 / && /--method (static|greedy|lru)/ { print (/triviaqa/ ? 16 : 10); exit }
    /triviaqa/ { print 8; exit }
    { print 6 }'
}
est_key() {   # estimate (GB) of class $1 with prior $2
  awk -F'\t' -v k="$1" -v pr="$2" '
    $1 == k && $3 == "ok" && $2 > ok { ok = $2 }
    $1 == k && $3 == "killed" && $2 > kd { kd = $2 }
    END { e = pr; if (1.2 * ok > e) e = 1.2 * ok; if (1.1 * kd > e) e = 1.1 * kd
          printf "%d\n", e + 0.999 }' "$PEAKS" 2>/dev/null || echo "$2"
}
est_of() { est_key "$(jkey "$1")" "$(prior_of "$1")"; }
priv_of() {   # private memory (GB, rounded up) of a pid
  awk '/^Private_(Clean|Dirty):/ {s += $2} END {printf "%d", s / 1048576 + 0.999}' /proc/$1/smaps_rollup 2>/dev/null
}
growth() {   # GB still expected to be allocated by registered jobs
  for f in $REG/*; do
    [ -f "$f" ] || continue
    pid=${f##*/}
    if [ ! -d /proc/$pid ]; then rm -f "$f"; continue; fi
    read k pr e < "$f"
    p=$(priv_of $pid); [ -n "$p" ] || continue
    [ "$e" -gt "$p" ] && echo $(( e - p ))
  done | awk -v f="$(cat ${RECAL_QUEUE_DIR:-/tmp/mx}/growth_factor 2>/dev/null || echo 1)" '{s += $1} END {printf "%d\n", f * s}'
}
mem() { awk -v w="$1" '/MemTotal/ {t = $2} /MemAvailable/ {a = $2} END {
  if (w == "used") printf "%d\n", (t - a) / 1048576; else printf "%d\n", a / 1048576 }' /proc/meminfo; }
same_running() {   # another live process is producing the same JSON target
  d=$(dirname "$JSON"); t=$(basename "$JSON" .json); tag=_${t#*_}
  ps -eo args | grep -v "sh -c" | grep -v mx_run_job | grep -q -- "--tag $tag --out $d\( \|$\)" && return 0
  ps -eo args | grep -v "sh -c" | grep -v mx_run_job | grep -q -- "--out $JSON\( \|$\)"
}
skipped() { [ -n "$JSON" ] && [ -f ${RECAL_QUEUE_DIR:-/tmp/mx}/skip.txt ] && grep -qxF "$JSON" ${RECAL_QUEUE_DIR:-/tmp/mx}/skip.txt; }
register() {   # record python descendants of $1 (two levels: sh -c -> python, sh -c -> sh -> python)
  for c in $(pgrep -P $1) $(for g in $(pgrep -P $1); do pgrep -P $g; done); do
    [ -f $REG/$c ] && continue
    case "$(cat /proc/$c/comm 2>/dev/null)" in python*) ;; *) continue ;; esac
    echo "$KEY $PRIO $EST" > $REG/$c
    [ "$PRIO" = "adhoc" ] && echo 1000 > /proc/$c/oom_score_adj 2>/dev/null || echo 300 > /proc/$c/oom_score_adj 2>/dev/null
  done
}

[ -n "$JSON" ] && [ -f "$JSON" ] && exit 0
skipped && exit 0
while [ -n "$JSON" ] && same_running; do sleep 60; done
[ -n "$JSON" ] && [ -f "$JSON" ] && exit 0
KEY=$(jkey "$LINE")
try=0
while :; do
  skipped && exit 0
  EST=$(est_of "$LINE")
  admitted=0
  exec 9> $LOCK
  if flock -w 30 9; then
    g=$(growth); u=$(mem used); a=$(mem avail)
    if [ ! -f ${RECAL_QUEUE_DIR:-/tmp/mx}/HOLD ] && [ $(( u + g + EST )) -le $CAP ] && [ $(( a - g - EST )) -ge $HEADROOM ]; then
      sh -c "$LINE" &
      job=$!
      for i in 1 2 3 4 5 6 7 8 9 10; do      # until its python process is registered (counted)
        register $job
        grep -lx "$KEY $PRIO $EST" $REG/* 2>/dev/null | while read f; do [ -d /proc/${f##*/} ] && echo y; done | grep -q y && break
        sleep 1
      done
      admitted=1
    fi
    flock -u 9
  fi
  exec 9>&-
  if [ $admitted -eq 0 ]; then
    sleep $((20 + $(od -An -N1 -tu1 /dev/urandom) % 40))
    continue
  fi
  peak=0
  while kill -0 $job 2>/dev/null; do
    register $job
    for c in $(pgrep -P $job) $(for g in $(pgrep -P $job); do pgrep -P $g; done); do
      p=$(priv_of $c); [ -n "$p" ] && [ "$p" -gt "$peak" ] && peak=$p
    done
    sleep 20
  done
  wait $job; rc=$?
  if { [ -n "$JSON" ] && [ -f "$JSON" ]; } || { [ -z "$JSON" ] && [ $rc -eq 0 ]; }; then
    [ "$peak" -gt 0 ] && printf '%s\t%s\tok\n' "$KEY" "$peak" >> $PEAKS
    exit 0
  fi
  if [ -n "$LOG" ] && grep -q "Traceback" "$LOG" 2>/dev/null; then
    echo "$(date +%T) FAILED (traceback) ${JSON:-$KEY}" >> ${RECAL_QUEUE_DIR:-/tmp/mx}/failures.log; exit 1
  fi
  if [ -z "$JSON" ] && [ $rc -le 128 ]; then
    echo "$(date +%T) FAILED (exit $rc) $KEY" >> ${RECAL_QUEUE_DIR:-/tmp/mx}/failures.log; exit 1
  fi
  [ "$peak" -gt 0 ] && printf '%s\t%s\tkilled\n' "$KEY" "$peak" >> $PEAKS
  try=$((try + 1))
  echo "$(date +%T) died without output (try $try, $PRIO, peak ${peak} GB, next estimate $(est_of "$LINE") GB) ${JSON:-$KEY}" >> ${RECAL_QUEUE_DIR:-/tmp/mx}/failures.log
  [ $try -ge 20 ] && { echo "$(date +%T) GAVE UP after $try attempts ${JSON:-$KEY}" >> ${RECAL_QUEUE_DIR:-/tmp/mx}/failures.log; exit 1; }
  b=$try; [ $b -gt 5 ] && b=5
  sleep $(( 60 * (1 << b) ))
done
