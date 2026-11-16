#!/bin/sh
# Stage-1 tuning (doc 9af): rerun with threaded index builds and the stage-2 reference = highest search list.
R=simulations/approximate_hits/results; S=simulations/approximate_hits
for P in spacev-aker-0.6 triviaqa-ip-0.6 esci1m; do
  S2=trace; [ $P = esci1m ] && S2=random
  for B in 0.01 0.1; do
    O=$R/timing_v4/$P/stage1_b$B.json
    mkdir -p $R/timing_v4/$P/logs $R/timing_v4/$P/stage1_first_version
    [ -f $O ] && mv $O $R/timing_v4/$P/stage1_first_version/
    .venv-ann/bin/python -u $S/stage1_tune.py --prep $R/online/$P --state $R/timing_v4/$P/state_b$B.npz --stream $S2 --threads 8 --out $O > $R/timing_v4/$P/logs/stage1_b$B.txt 2>&1 &
    sleep 2
  done
done
wait
touch /tmp/mx/STAGE1_REDO.DONE
