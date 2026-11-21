#!/bin/sh
# Watchdog for the matrix driver (run_matrix.sh, copied to /tmp/mx/driver3.sh).
# Every 10 s: if MemAvailable < 15 GB (emergency only; mx_run_job.sh admits by committed memory), kill the most recently started simulator /
# real-Aker job (it has lost the least work; mx_run_job.sh retries it later)
# instead of letting the kernel's out-of-memory killer pick a long-running one.
# Light jobs get oom_score_adj 800, heavy 500, so the kernel kills them before a prep step.
# Every 10 min: append a status line to /tmp/mx/status.log and restart the driver
# if it is gone before /tmp/mx/ALL_3.DONE exists (safe: finished and running jobs
# are skipped by mx_run_job.sh). Every hour: mx_requeue.sh for given-up jobs.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
i=0
while [ ! -f $M/HX.DONE ] || [ ! -f $M/DISKANN_TQA_IP_PQ.DONE ]; do
  # make the kernel's out-of-memory killer prefer light jobs (ESCI, 1% budget, real Aker;
  # cheap to retry) over the long heavy ones (ours / LRU at a 10% budget)
  # heavy jobs 500, light jobs 800; prep steps (one at a time, hours each) keep 0
  ps -eo pid,args | grep -E "^ *[0-9]+ [^ ]*python[^ ]* -u [^ ]*(online_sim.py|eval_aker_real.py)" > $M/.jobs
  for pid in $(grep -vE -- "--budget 0\.1 .*--method (static|greedy|lru)" $M/.jobs | awk '{print $1}'); do
    [ "$(cat /proc/$pid/oom_score_adj 2>/dev/null)" = "800" ] || echo 800 > /proc/$pid/oom_score_adj 2>/dev/null
  done
  for pid in $(grep -E -- "--budget 0\.1 .*--method (static|greedy|lru)" $M/.jobs | awk '{print $1}'); do
    [ "$(cat /proc/$pid/oom_score_adj 2>/dev/null)" = "500" ] || echo 500 > /proc/$pid/oom_score_adj 2>/dev/null
  done
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if [ "$avail" -lt 15 ]; then
    # youngest job = smallest elapsed seconds
    # youngest light job first; the youngest heavy job only if no light job runs
    jobs=$(ps -eo pid,etimes,args | grep -E "^ *[0-9]+ +[0-9]+ [^ ]*python[^ ]* -u [^ ]*(online_sim.py|eval_aker_real.py)")
    pid=$(printf '%s\n' "$jobs" | grep -vE -- "--budget 0\.1 .*--method (static|greedy|lru)" | sort -k2 -n | head -1 | awk '{print $1}')
    [ -n "$pid" ] || pid=$(printf '%s\n' "$jobs" | sort -k2 -n | head -1 | awk '{print $1}')
    if [ -n "$pid" ]; then
      echo "$(date +%T) low memory (${avail} GB): killed youngest job $pid: $(ps -o args= -p $pid | cut -c1-200)" >> $M/watchdog.log
      kill -9 $pid
    fi
  fi
  if [ $((i % 60)) -eq 0 ]; then
    sims=$(ps -eo args | grep -cE "^[^ ]*python[^ ]* -u [^ ]*online_sim.py")
    akers=$(ps -eo args | grep -cE "^[^ ]*python[^ ]* -u [^ ]*eval_aker_real.py")
    fails=$(grep -c "" $M/failures.log 2>/dev/null); fails=${fails:-0}
    gaveup=$(grep -c "GAVE UP\|traceback" $M/failures.log 2>/dev/null); gaveup=${gaveup:-0}
    njson=$(find simulations/approximate_hits/results/online simulations/approximate_hits/results/aker_real -name "*.json" -newer $M/hx.sh | wc -l)
    echo "$(date '+%m-%d %T') free ${avail} GB; running sims $sims, aker $akers; new results since launch $njson; retries $fails, gave up / traceback $gaveup" >> $M/status.log
    # supervised launchers (all safe to restart: done / running jobs are skipped)
    for pair in hx.sh:HX.DONE diskann_tqa_ip_pq.sh:DISKANN_TQA_IP_PQ.DONE aker_tuned_ip.sh:AKER_TUNED_IP.DONE aker_tuned_ip2.sh:AKER_TUNED_IP2.DONE after_timing.sh:HX.DONE; do
      sc=${pair%%:*}; fl=${pair#*:}
      [ -f $M/$sc ] || continue
      if [ ! -f $M/$fl ] && ! pgrep -f "${RECAL_QUEUE_DIR:-/tmp/mx}/$sc" > /dev/null; then
        echo "$(date +%T) $sc not running: restarting" >> $M/watchdog.log
        nohup sh ${RECAL_QUEUE_DIR:-/tmp/mx}/$sc >> $M/${sc%.sh}.log 2>&1 &
      fi
    done
  fi
  # Aker's telemetry dumps (/tmp is RAM-backed): remove any older than 10 min
  if [ $((i % 30)) -eq 0 ]; then
    find /tmp -maxdepth 1 -name "aker_trace_*" -type d -mmin +10 -exec rm -rf {} + 2>/dev/null
  fi
  # every hour: re-queue given-up jobs (in the background)
  # (hourly re-queue of old lists disabled 11-12: only the plan's lists run now)
  i=$((i + 1))
  sleep 10
done
echo "$(date +%T) all done" >> $M/watchdog.log
