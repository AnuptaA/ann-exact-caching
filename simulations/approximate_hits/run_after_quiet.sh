#!/bin/sh
# doc 9ai: after the quiet checks: headline per-request reruns (graph path, priority order) and pool
# statistics side by side; interim plots every hour, final plots at the end.
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx; S=simulations/approximate_hits
plots() {
  TIMING_VERSION=timing_v6 PYTHONPATH=. .venv/bin/python $S/plot_timing.py > /dev/null 2>&1
  HEADLINE_V6=1 PYTHONPATH=. .venv/bin/python $S/plot_headline.py > /dev/null 2>&1
  echo "$(date +%T) plots redrawn"
}
while [ ! -f $M/QUIET_CHECKS.DONE ]; do sleep 60; done
echo "$(date +%T) headline + pool"
python3 $S/run_jobs.py $M/v6_headline_g.txt --workers 8 --cores-per-job 4 --first-core 8 --first 0.01,0.1 > $M/v6_headline_g.log 2>&1 &
H=$!
python3 $S/run_jobs.py $M/v6_pool_g.txt --workers 6 --cores-per-job 4 --first-core 48 > $M/v6_pool_g.log 2>&1 &
P=$!
while kill -0 $H 2>/dev/null || kill -0 $P 2>/dev/null; do
  sleep 3600
  plots
done
plots
echo "$(date +%T) done"
touch $M/AFTER_QUIET.DONE
