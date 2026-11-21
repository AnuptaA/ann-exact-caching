#!/bin/sh
# Backend-recall variant of a workload as an alias (doc 9af / 9ae d): same data, DiskANN at a higher
# recall target: answers, base node cache (the source's), choice.json, warm-up states, stage-1 tuning,
# quiet arm, composed runs, plot. Usage:
#   sh run_recall_variant.sh <source prep> <alias> <target> <search list> <beam> <stream> "<budgets>"
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
SRC=$1; P=$2; TGT=$3; L=$4; W=$5; STREAM=$6; BUDGETS=$7
M=${RECAL_QUEUE_DIR:-/tmp/mx}; S=simulations/approximate_hits; R=$S/results; B=$R/backends/$P; O=$R/timing_v4/$P
export PYTHONPATH=.
ln -sfn $SRC $R/online/$P
[ -d $R/aker_real/$SRC ] && ln -sfn $SRC $R/aker_real/$P
IDX=$(.venv/bin/python -c "import json; print(json.load(open('$R/backends/$SRC/diskann.json')).get('index_dir') or '$R/backends/$SRC/diskann')")
mkdir -p $B/logs $O/logs
SHARE=""; [ $L -ge 500 ] && SHARE="--share-deep"
[ -f $B/diskann.json ] || OMP_NUM_THREADS=24 .venv-ann/bin/python -u $S/eval_real_backend.py --prep $R/online/$SRC --index diskann \
    --out-name $P --threads 24 --index-dir $IDX --fixed $TGT:$L --beam $W $SHARE > $B/logs/answers.txt 2>&1
[ -f $B/diskann.json ] || { echo "answers failed"; exit 1; }
echo "$(date +%T) answers ready"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cp -n $R/backends/$SRC/node_cache.json $B/node_cache.json
.venv/bin/python - <<PY
import json
d = json.load(open("$B/diskann.json"))
t = d["targets"]["$TGT"]
json.dump({"target": float("$TGT"), "index_dir": d["index_dir"], "L": t["effort"], "W": d["beam"],
           "recall": t["recall_answer_at_k"], "recall_source": "all queries (eval_real_backend --fixed)",
           "source_prep": "$SRC"}, open("$B/choice.json", "w"), indent=1)
print(t)
PY
for b in $BUDGETS; do
  [ -f $O/state_b$b.json ] || .venv/bin/python -u $S/online_sim.py --prep $R/online/$P --method greedy --detect --stream $STREAM --budget $b --max-queries 21000 --shell-scope balls --shell-balls 10 --backend-file $B/diskann.npz --backend-target $TGT --save-state $O/state_b$b.npz --tag _state_b$b --out $O > $O/logs/state_b$b.txt 2>&1 && ln -sf greedy_state_b$b.json $O/state_b$b.json &
done
wait
echo "$(date +%T) states: $(ls $O/state_b*.npz | wc -l)"
sh $S/run_stage1_all.sh
echo "$(date +%T) tuning: $(ls $O/stage1_b*.json | wc -l)"
.venv/bin/python $S/timing_jobs.py --set v5 | grep "eval_timing_backend.py" | grep "/$P/" > $M/arm_$P.txt
while IFS= read -r l; do echo "$(date +%T) arm"; sh -c "$l"; done < $M/arm_$P.txt
echo "$(date +%T) composed runs"
.venv/bin/python $S/run_timing_composed.py --workers 12 --set v5 --version timing_v5
TIMING_VERSION=timing_v5 .venv/bin/python $S/plot_timing.py
python3 $S/v5_summary.py > $R/latest/timing_v5_summary.txt
echo "$(date +%T) done"
touch $M/VARIANT_$P.DONE
