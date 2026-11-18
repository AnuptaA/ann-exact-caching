#!/bin/sh
# doc 9ai: timing-sensitive checks once nothing else runs: solo LRU cells (one at a time), then the TripClick
# runs of the new core (packed codes; nearest balls) 4 at a time; then the timing plot.
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx; S=simulations/approximate_hits
while [ ! -f $M/V6_AFTER.DONE ] || [ ! -f $M/V6_POOL.DONE ]; do sleep 60; done
echo "$(date +%T) solo checks"
python3 $S/run_jobs.py $M/v6c_solo.txt --workers 1 --first-core 8
echo "$(date +%T) TripClick runs"
python3 $S/run_jobs.py $M/v6c.txt --workers 4 --first-core 8
echo "$(date +%T) done"
touch $M/QUIET_CHECKS.DONE
