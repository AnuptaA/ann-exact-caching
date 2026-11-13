#!/bin/sh
# Headline runs for a NEW trace workload (11-13: tripclick-ip, medrag-ip), the same job set as
# TriviaQA ip Zipf 0.6: every job line that ran for triviaqa-ip-0.6 (ours / LRU at the matrix
# configurations, the Aker / QVCache grids, real Aker at the paper and tuned settings; cache sizes
# 0.3 / 1 / 3 / 10%), the latest line per output JSON (later lists supersede earlier ones, e.g. the
# fit-start re-runs), with the workload name substituted.
#   run_headline_new_workload.sh <prep>      (waits for results/online/<prep>/top500.npy)
# Template lists: /tmp/mx job lists (see the list order below). Flag: /tmp/mx/HEADLINE_<prep>.DONE
P=$1
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MX_PRIORITY=plot
export LD_LIBRARY_PATH=/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:/home/jialind/third_party/faiss-install/lib64
M=/tmp/mx
S=simulations/approximate_hits
while [ ! -f $S/results/online/$P/top500.npy ]; do sleep 300; done
TPL=triviaqa-ip-0.6
for f in H_tqa_ip_0.6 HA_tqa_ip_0.6 aker_tuned_ip aker_tuned_ip2 HX HXA HR; do
  [ -f $M/$f.txt ] && grep -E "(online|aker_real)/$TPL[ /]" $M/$f.txt
done | awk '{ if (match($0, /\[ -f [^ ]*json/)) k = substr($0, RSTART, RLENGTH); else k = $0
              if (!(k in last)) order[++n] = k; last[k] = $0 }
            END { for (i = 1; i <= n; i++) print last[order[i]] }' \
  | sed "s|$TPL|$P|g" > $M/H_new_$P.all
grep -v "eval_aker_real.py" $M/H_new_$P.all > $M/H_new_$P.txt
grep "eval_aker_real.py" $M/H_new_$P.all > $M/HA_new_$P.txt
echo "$(date +%T) $P: $(grep -c . $M/H_new_$P.txt) simulator jobs, $(grep -c . $M/HA_new_$P.txt) real-Aker jobs"
sh $S/mx_complete.sh $M/H_new_$P.txt $M/H_new_$P.C 24 &
sh $S/mx_complete.sh $M/HA_new_$P.txt $M/HA_new_$P.C 8 &
wait
touch $M/HEADLINE_$P.DONE
echo "$(date +%T) $P headline done"
