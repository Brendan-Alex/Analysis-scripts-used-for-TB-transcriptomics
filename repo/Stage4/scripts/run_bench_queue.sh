#!/bin/bash
# runs pending benchmark jobs within one call's time budget (resumable via results/bench/<task>__done.txt)
cd "$(dirname "$0")/.."
START=$(date +%s)
for T in ${TASKS_TO_RUN:-LTB_vs_CON ATB_vs_OD ATB_vs_LTB}; do
  LEFT=$(( 160 - ($(date +%s) - START) )); [ $LEFT -lt 30 ] && { echo "stop: time"; exit 0; }
  python3 scripts/02_bench.py run $T $(( LEFT - 25 )) 2>&1 | grep -v -E "RuntimeWarning|stddev|warn" | tail -2
done
