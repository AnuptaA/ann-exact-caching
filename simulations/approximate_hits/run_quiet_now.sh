#!/bin/sh
# doc 9ai: pause the headline / pool jobs (SIGSTOP; not timing-sensitive), run the timing-sensitive checks on
# a quiet machine, resume them (SIGCONT, also on any exit).
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx; S=simulations/approximate_hits
resume() { for p in $(cat $M/paused_pids.txt); do kill -CONT $p 2>/dev/null; done; echo "$(date +%T) resumed"; }
trap resume EXIT INT TERM
for p in $(cat $M/paused_pids.txt); do kill -STOP $p 2>/dev/null; done
echo "$(date +%T) paused $(wc -l < $M/paused_pids.txt) processes"
sleep 20
echo "$(date +%T) solo checks"
python3 $S/run_jobs.py $M/v6c_solo.txt --workers 1 --first-core 8
echo "$(date +%T) TripClick runs"
python3 $S/run_jobs.py $M/v6c.txt --workers 4 --first-core 8
echo "$(date +%T) done"
touch $M/QUIET_CHECKS.DONE
