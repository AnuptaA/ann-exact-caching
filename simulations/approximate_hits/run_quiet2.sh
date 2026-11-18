#!/bin/sh
# doc 9ak: quiet window for the TripClick fast-path / RaBitQ / fewer-samples timing runs (timing_v6c). Pauses the
# untimed queues (runners and all their descendants, SIGSTOP; nothing new can start), runs /tmp/mx/v6d.txt 4 at
# a time, resumes them (SIGCONT, also on any exit).
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx; S=simulations/approximate_hits
desc() { for c in $(pgrep -P "$1"); do echo "$c"; desc "$c"; done; }
resume() { for p in $(cat $M/paused2_pids.txt 2>/dev/null); do kill -CONT "$p" 2>/dev/null; done; echo "$(date +%T) resumed"; }
trap resume EXIT INT TERM
# wait for the untimed smoke runs (they must not overlap the timed ones)
while pgrep -f 'run_jobs.py /tmp/mx/smoke9ak' > /dev/null; do sleep 30; done
: > $M/paused2_pids.txt
for r in $(pgrep -f 'run_jobs.py /tmp/mx/(v6_twins|attrib|v6_pool|v6_headline)'); do
  echo "$r" >> $M/paused2_pids.txt
  desc "$r" >> $M/paused2_pids.txt
done
for p in $(cat $M/paused2_pids.txt); do kill -STOP "$p" 2>/dev/null; done
echo "$(date +%T) paused $(wc -l < $M/paused2_pids.txt) processes"
sleep 20
python3 $S/run_jobs.py $M/v6d.txt --workers 4 --first-core 8 --first 0.1,0.01,0.03,0.003
echo "$(date +%T) done"
touch $M/QUIET2.DONE
