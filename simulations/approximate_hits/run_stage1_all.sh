#!/bin/sh
# Stage-1 / stage-2 search lists (fp32) for every cache state of the timing batch (doc 9af); 8 at a time.
R=simulations/approximate_hits/results; S=simulations/approximate_hits
for f in $R/timing_v4/*/state_b*.npz; do
  P=$(basename $(dirname $f)); B=$(basename $f .npz | sed 's/state_b//')
  case "$P:$B" in medrag-ip:0.01|medrag-ip:0.03|medrag-ip:0.1) continue;; esac
  S2=trace; case $P in esci*) S2=random;; esac
  TR=$(python3 -c "import json; print(f\"{json.load(open('$R/backends/$P/choice.json'))['target']:g}\")" 2>/dev/null || echo 0.95)
  O=$R/timing_v4/$P/stage1_b$B.json
  [ -f $O ] && continue
  mkdir -p $R/timing_v4/$P/logs
  echo ".venv-ann/bin/python -u $S/stage1_tune.py --prep $R/online/$P --state $f --stream $S2 --threads 4 --precisions fp32 --target-recall $TR --out $O > $R/timing_v4/$P/logs/stage1_b$B.txt 2>&1"
done | xargs -P 8 -I{} sh -c "{}"
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/STAGE1_ALL.DONE
