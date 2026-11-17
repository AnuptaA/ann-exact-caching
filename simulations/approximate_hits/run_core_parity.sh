#!/bin/sh
# doc 9ah milestone 2: online_sim with --cache-impl python vs core, TripClick 1%, first 20,000 requests
# after warm-up (greedy + rebuild, LRU, continuous greedy; once per batch), after the quiet node-cache curves.
cd /local/home/jialind/ann-exact-caching || exit 1
R=simulations/approximate_hits/results; O=/tmp/mx/par2
while [ ! -f /tmp/mx/BENCH_CURVES.DONE ]; do sleep 60; done
mkdir -p $O
for impl in python core; do
  PYTHONPATH=. .venv-core/bin/python -u simulations/approximate_hits/online_sim.py --prep $R/online/tripclick-ip --method greedy --response rebuild --stream trace --budget 0.01 --seed 0 --detect --max-queries 40000 --cache-impl $impl --out $O/greedy_$impl > $O/greedy_$impl.txt 2>&1 &
  PYTHONPATH=. .venv-core/bin/python -u simulations/approximate_hits/online_sim.py --prep $R/online/tripclick-ip --method lru --k-lru 10 --stream trace --budget 0.01 --seed 0 --detect --max-queries 40000 --cache-impl $impl --out $O/lru_$impl > $O/lru_$impl.txt 2>&1 &
  PYTHONPATH=. .venv-core/bin/python -u simulations/approximate_hits/online_sim.py --prep $R/online/tripclick-ip --method ogreedy --og-evict utility --stream trace --budget 0.01 --seed 0 --detect --max-queries 40000 --cache-impl $impl --out $O/ogreedy_$impl > $O/ogreedy_$impl.txt 2>&1 &
done
# per-request admission, core only (the python cache would take ~20 h)
PYTHONPATH=. .venv-core/bin/python -u simulations/approximate_hits/online_sim.py --prep $R/online/tripclick-ip --method lru --k-lru 10 --stream trace --budget 0.01 --seed 0 --detect --update-every 1 --cache-impl core --out $O/lru_u1_core > $O/lru_u1_core.txt 2>&1 &
wait
touch /tmp/mx/CORE_PARITY.DONE
