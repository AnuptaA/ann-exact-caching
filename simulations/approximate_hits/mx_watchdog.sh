#!/bin/sh
# Watchdog for the matrix driver (run_matrix.sh, copied to /tmp/mx/driver3.sh).
# Every 20 s: if MemAvailable < 20 GB, kill the most recently started simulator /
# real-Aker job (it has lost the least work; mx_run_job.sh retries it later)
# instead of letting the kernel's out-of-memory killer pick a long-running one.
# Every 10 min: append a status line to /tmp/mx/status.log and restart the driver
# if it is gone before /tmp/mx/ALL_3.DONE exists (safe: finished and running jobs
# are skipped by mx_run_job.sh).
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx
i=0
while [ ! -f $M/ALL_3.DONE ]; do
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if [ "$avail" -lt 20 ]; then
    # youngest job = smallest elapsed seconds
    pid=$(ps -eo pid,etimes,args | grep -v "sh -c" | grep -v mx_run_job | grep -E "^ *[0-9]+ +[0-9]+ [^ ]*python[^ ]* -u [^ ]*(online_sim.py|eval_aker_real.py)" \
          | sort -k2 -n | head -1 | awk '{print $1}')
    if [ -n "$pid" ]; then
      echo "$(date +%T) low memory (${avail} GB): killed youngest job $pid: $(ps -o args= -p $pid | cut -c1-200)" >> $M/watchdog.log
      kill -9 $pid
    fi
  fi
  if [ $((i % 30)) -eq 0 ]; then
    sims=$(ps -eo args | grep -cE "^[^ ]*python[^ ]* -u [^ ]*online_sim.py")
    akers=$(ps -eo args | grep -cE "^[^ ]*python[^ ]* -u [^ ]*eval_aker_real.py")
    fails=$(grep -c "" $M/failures.log 2>/dev/null); fails=${fails:-0}
    gaveup=$(grep -c "GAVE UP\|traceback" $M/failures.log 2>/dev/null); gaveup=${gaveup:-0}
    njson=$(find simulations/approximate_hits/results/online simulations/approximate_hits/results/aker_real -name "*.json" -newer $M/driver.sh | wc -l)
    echo "$(date '+%m-%d %T') free ${avail} GB; running sims $sims, aker $akers; new results since launch $njson; retries $fails, gave up / traceback $gaveup" >> $M/status.log
    if ! pgrep -f "/tmp/mx/driver3.sh" > /dev/null; then
      echo "$(date +%T) driver3 not running: restarting" >> $M/watchdog.log
      nohup /tmp/mx/driver3.sh >> $M/driver3.log 2>&1 &
    fi
  fi
  i=$((i + 1))
  sleep 20
done
echo "$(date +%T) all done" >> $M/watchdog.log
