#!/bin/sh
# Re-queue matrix jobs that mx_run_job.sh gave up on (died without output 3 times,
# e.g. out-of-memory kills during a memory crunch), at most twice per job overall
# (/tmp/mx/requeued.log). Job lines are looked up in the queue files of the drivers.
# Run by mx_watchdog.sh every hour; safe to run by hand.
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx
W=simulations/approximate_hits/mx_run_job.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export LD_LIBRARY_PATH=/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:/home/jialind/third_party/faiss-install/lib64:$LD_LIBRARY_PATH
touch $M/requeued.log
OUT=$M/requeue_$(date +%H%M%S).txt
: > $OUT
for j in $(grep "GAVE UP" $M/failures.log 2>/dev/null | awk '{print $NF}' | sort -u); do
  [ -f "$j" ] && continue
  [ "$(grep -cxF "$j" $M/requeued.log)" -ge 2 ] && continue
  line=$(cat $M/H_3.txt $M/L1_3.txt $M/L10_3.txt $M/A_3.txt $M/D_real.txt $M/AD_real.txt $M/T_*.txt $M/TA_*.txt 2>/dev/null \
         | grep -F "[ -f $j ]" | head -1)
  [ -n "$line" ] || continue
  printf '%s\n' "$j" >> $M/requeued.log
  printf '%s\n' "$line" >> $OUT
done
[ -s $OUT ] || { rm -f $OUT; exit 0; }
echo "$(date +%T) re-queued $(grep -c '' $OUT) jobs ($OUT)" >> $M/watchdog.log
grep -E -- "--budget 0\.1 .*--method (static|greedy|lru)|triviaqa.*--method (static|greedy|lru)" $OUT > $OUT.H
grep -vE -- "--budget 0\.1 .*--method (static|greedy|lru)|triviaqa.*--method (static|greedy|lru)" $OUT > $OUT.L
( xargs -P 3 -I{} sh $W 70 {} < $OUT.H ) &
( xargs -P 6 -I{} sh $W 45 {} < $OUT.L ) &
wait
