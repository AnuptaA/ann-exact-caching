#!/bin/sh
# index study: every remaining build through the runner as ad-hoc work (retried after kills, from a
# clean index folder); mips only if its current (unmanaged) build ends without a result
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=16 MX_PRIORITY=adhoc
C=simulations/approximate_hits/mx_complete.sh
L=simulations/approximate_hits/results/backends/index_study
( sh $C ${RECAL_QUEUE_DIR:-/tmp/mx}/index_study_all.txt ${RECAL_QUEUE_DIR:-/tmp/mx}/IS_ALL_C.DONE 2 ) &
( while pgrep -f "index_study.py --variant mips" > /dev/null; do sleep 120; done
  sh $C ${RECAL_QUEUE_DIR:-/tmp/mx}/index_study_mips.txt ${RECAL_QUEUE_DIR:-/tmp/mx}/IS_MIPS_C.DONE 1 ) &
wait
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/INDEX_STUDY.DONE
