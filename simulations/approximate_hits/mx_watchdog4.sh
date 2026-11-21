#!/bin/sh
# Watchdog v4 (11-13): memory emergency kills by PRIORITY, launcher supervision, status line.
# Usage: mx_watchdog4.sh <flag>...   (runs until every flag exists)
# Below 15 GB available it kills ONE process, chosen in this order:
#   1. a registered ad-hoc job (/tmp/mx/registry, priority adhoc), youngest first;
#   2. an unregistered python process of ours with >= 2 GB private memory (by definition ad hoc:
#      not started by the runner, e.g. a hand-run check), youngest first;
#   3. a registered plot job, youngest first (the runner retries it later).
# Processes registered with priority "timing" (the quiet timing runs) are never killed.
# Admission control loop: each kill raises the growth-reservation factor (/tmp/mx/growth_factor)
# by 0.25 (max 1); each 15 min without kills and with >= 60 GB available lowers it by 0.1 (min 0.2).
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
REG=$M/registry
mkdir -p $REG
priv_of() { awk '/^Private_(Clean|Dirty):/ {s += $2} END {printf "%d", s / 1048576}' /proc/$1/smaps_rollup 2>/dev/null; }
prio_of() { [ -f $REG/$1 ] && awk '{print $2}' $REG/$1 || echo none; }
victim() {
  me=$(id -u)
  ps -u "$me" -o pid=,etimes=,comm= | awk '$3 ~ /^python/ {print $1, $2}' | sort -k2 -n > $M/.cand
  for want in adhoc none plot; do
    while read pid age; do
      p=$(prio_of $pid)
      [ "$p" = "$want" ] || continue
      if [ "$want" = none ]; then pv=$(priv_of $pid); [ -n "$pv" ] && [ "$pv" -ge 2 ] || continue; fi
      echo $pid; return
    done < $M/.cand
  done
}
i=0
while :; do
  done_all=1
  for f in "$@"; do [ -f $M/$f ] || done_all=0; done
  [ $done_all -eq 1 ] && break
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if [ "$avail" -lt 15 ]; then
    pid=$(victim)
    if [ -n "$pid" ]; then
      echo "$(date +%T) low memory (${avail} GB): killed $(prio_of $pid) job $pid: $(ps -o args= -p $pid | cut -c1-200)" >> $M/watchdog.log
      kill -9 $pid
      # adapt admission: reserve more of the registered jobs' expected growth
      awk '{f = $1 + 0.25; if (f > 1) f = 1; printf "%.2f\n", f}' $M/growth_factor > $M/growth_factor.new 2>/dev/null &&
        mv $M/growth_factor.new $M/growth_factor
      last_kill=$i
    fi
  fi
  # 15 min (90 x 10 s) without a kill and with >= 60 GB available: reserve less
  if [ $(( i - ${last_kill:-0} )) -ge 90 ] && [ $((i % 90)) -eq 0 ] && [ "$avail" -ge 60 ]; then
    awk '{f = $1 - 0.1; if (f < 0.2) f = 0.2; printf "%.2f\n", f}' $M/growth_factor > $M/growth_factor.new 2>/dev/null &&
      mv $M/growth_factor.new $M/growth_factor
  fi
  if [ $((i % 60)) -eq 0 ]; then
    for f in $REG/*; do [ -f "$f" ] && [ ! -d /proc/${f##*/} ] && rm -f "$f"; done
    n_plot=$(cat $REG/* 2>/dev/null | awk '$2 == "plot"' | wc -l)
    n_adhoc=$(cat $REG/* 2>/dev/null | awk '$2 == "adhoc"' | wc -l)
    echo "$(date '+%m-%d %T') free ${avail} GB; growth factor $(cat $M/growth_factor); registered plot $n_plot, adhoc $n_adhoc; retries $(grep -c 'died without' $M/failures.log 2>/dev/null), gave up $(grep -c 'GAVE UP' $M/failures.log 2>/dev/null)" >> $M/status.log
    for pair in req2.sh:REQ.DONE he.sh:HE.DONE es.sh:TSTATES_ESCI.DONE after_timing.sh:TIMING3C.DONE; do
      sc=${pair%%:*}; fl=${pair#*:}
      [ -f $M/$sc ] || continue
      if [ ! -f $M/$fl ] && ! pgrep -f "${RECAL_QUEUE_DIR:-/tmp/mx}/$sc" > /dev/null; then
        echo "$(date +%T) $sc not running: restarting" >> $M/watchdog.log
        nohup sh ${RECAL_QUEUE_DIR:-/tmp/mx}/$sc >> $M/${sc%.sh}.log 2>&1 &
      fi
    done
  fi
  if [ $((i % 30)) -eq 0 ]; then
    find /tmp -maxdepth 1 -name "aker_trace_*" -type d -mmin +10 -exec rm -rf {} + 2>/dev/null
  fi
  i=$((i + 1))
  sleep 10
done
echo "$(date +%T) watchdog v4 done" >> $M/watchdog.log
