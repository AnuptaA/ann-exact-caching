#!/bin/sh
# index study, second phase through the runner as ad-hoc work; done when all five results exist
cd /local/home/jialind/ann-exact-caching || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=16 MX_PRIORITY=adhoc
L=simulations/approximate_hits/results/backends/index_study
while [ ! -f $L/pq768/result.json ] || [ ! -f $L/mips/result.json ]; do sleep 120; done
sh simulations/approximate_hits/mx_complete.sh /tmp/mx/index_study2.txt /tmp/mx/INDEX_STUDY2_C.DONE 2
for v in base pq768 mips r128 mem; do [ -f $L/$v/result.json ] || echo "missing $v" >> /tmp/mx/index_study2.log; done
touch /tmp/mx/INDEX_STUDY.DONE
