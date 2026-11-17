#!/bin/sh
# SPACEV real log with DiskANN tuned to recall 0.95 (search list 2000, beam 16), as workload alias
# spacev-real95 (doc 9af): same data (online/spacev-real95 -> spacev-real), own answers / states / runs.
# Waits for the answers (eval_real_backend --out-name spacev-real95), then: node cache (spacev-real's
# choice, same index), choice.json, warm-up states, stage-1 tuning, quiet arm, composed runs, plot.
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx; S=simulations/approximate_hits; R=$S/results; P=spacev-real95; B=$R/backends/$P; O=$R/timing_v4/$P
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
while [ ! -f $B/diskann.json ]; do sleep 120; done
echo "$(date +%T) answers ready"
cp -n $R/backends/spacev-real/node_cache.json $B/node_cache.json
.venv/bin/python - <<PY
import json
d = json.load(open("$B/diskann.json"))
t = d["targets"]["0.95"]
json.dump({"target": 0.95, "index_dir": d["index_dir"], "L": t["effort"], "W": d["beam"],
           "recall": t["recall_answer_at_k"], "recall_source": "all queries (eval_real_backend --fixed)"},
          open("$B/choice.json", "w"), indent=1)
print(t)
PY
mkdir -p $O/logs
for b in 0.01 0.1; do
  [ -f $O/state_b$b.json ] || .venv/bin/python -u $S/online_sim.py --prep $R/online/$P --method greedy --detect --stream trace --budget $b --max-queries 21000 --shell-scope balls --shell-balls 10 --backend-file $B/diskann.npz --backend-target 0.95 --save-state $O/state_b$b.npz --tag _state_b$b --out $O > $O/logs/state_b$b.txt 2>&1 && ln -sf greedy_state_b$b.json $O/state_b$b.json &
done
wait
echo "$(date +%T) states: $(ls $O/state_b*.npz | wc -l)"
.venv-ann/bin/python -c "
import numpy as np
for b in ('0.01', '0.1'):
    print(b, 'cutoff', np.load('$O/state_b' + b + '.npz')['lam'])"
sh $S/run_stage1_all.sh
echo "$(date +%T) tuning: $(ls $O/stage1_b*.json | wc -l)"
.venv/bin/python $S/timing_jobs.py --set v5 | grep "eval_timing_backend.py" | grep "/$P/" > $M/r95_arm.txt
while IFS= read -r l; do echo "$(date +%T) arm"; sh -c "$l"; done < $M/r95_arm.txt
echo "$(date +%T) composed runs"
.venv/bin/python $S/run_timing_composed.py --workers 12 --set v5 --version timing_v5
TIMING_VERSION=timing_v5 .venv/bin/python $S/plot_timing.py
python3 $S/v5_summary.py > $R/latest/timing_v5_summary.txt
echo "$(date +%T) done"
touch $M/SPACEV_REAL95.DONE
