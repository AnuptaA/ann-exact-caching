#!/bin/sh
# doc 9ah: after the v6 timing batch: (1) its failed jobs again (Aker library paths fixed; finished runs are
# skipped), (2) contention check: the two heaviest LRU cells alone, (3) headline per-request reruns, (4) plots.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
S=simulations/approximate_hits; R=$S/results; M=${RECAL_QUEUE_DIR:-/tmp/mx}
while [ ! -f $M/V6_TIMING.DONE ]; do sleep 60; done
echo "$(date +%T) timing rerun of failed jobs"
python3 $S/run_jobs.py $M/v6_timing2.txt --workers 8
echo "$(date +%T) contention check (alone)"
for c in "tripclick-ip 0.1" "spacev-aker-0.99 0.1"; do
  set -- $c
  l=$(grep "timing_v6/$1/lru_b$2 " $M/v6_timing2.txt | head -1 | sed "s|--out [^ ]*|--out $R/timing_v6/$1/alone_lru_b$2|; s|\[ -f [^]]*\] || ||; s|> [^ ]*logs/lru_b$2.txt|> $R/timing_v6/$1/logs/alone_lru_b$2.txt|")
  sh -c "$(echo "$l" | sed 's|.venv-core/bin/python|taskset -c 8-11 .venv-core/bin/python|')"
done
echo "$(date +%T) headline reruns"
PYTHONPATH=. .venv/bin/python $S/jobs_v6.py --set headline > $M/v6_headline.txt
python3 $S/run_jobs.py $M/v6_headline.txt --workers 8 --cores-per-job 8 --first-core 8 --first 0.01,0.1,0.003,0.03
echo "$(date +%T) plots"
TIMING_VERSION=timing_v6 PYTHONPATH=. .venv/bin/python $S/plot_timing.py
HEADLINE_V6=1 PYTHONPATH=. .venv/bin/python $S/plot_headline.py
echo "$(date +%T) done"
touch $M/V6_AFTER.DONE
