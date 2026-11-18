#!/bin/sh
# doc 9al: quiet window for the full f 0.95 timing rerun (timing_v6/*/f95*), 8 at a time as the v6 batch. Pauses the
# untimed queues (runners and all their descendants, SIGSTOP; nothing new can start), runs /tmp/mx/v6d.txt 4 at
# a time, resumes them (SIGCONT, also on any exit).
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx; S=simulations/approximate_hits
desc() { for c in $(pgrep -P "$1"); do echo "$c"; desc "$c"; done; }
resume() { for p in $(cat $M/paused3_pids.txt 2>/dev/null); do kill -CONT "$p" 2>/dev/null; done; echo "$(date +%T) resumed"; }
trap resume EXIT INT TERM
# wait for the untimed smoke runs (they must not overlap the timed ones)
while [ ! -f $M/QUIET2.DONE ]; do sleep 60; done; sleep 30
: > $M/paused3_pids.txt
for r in $(pgrep -f 'run_jobs.py /tmp/mx/(attrib|v6_pool|v6_headline|graphef)'); do
  echo "$r" >> $M/paused3_pids.txt
  desc "$r" >> $M/paused3_pids.txt
done
for p in $(cat $M/paused3_pids.txt); do kill -STOP "$p" 2>/dev/null; done
echo "$(date +%T) paused $(wc -l < $M/paused3_pids.txt) processes"
sleep 20
python3 $S/run_jobs.py $M/v6f95.txt --workers 8 --first-core 8 --first 0.01,0.1
echo "$(date +%T) done"
touch $M/QUIET3.DONE
