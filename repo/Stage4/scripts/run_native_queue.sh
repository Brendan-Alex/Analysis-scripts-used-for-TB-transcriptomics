#!/bin/bash
# runs pending Native folds within one call's time budget (each fold resumes from its checkpoint)
cd "$(dirname "$0")/.."
START=$(date +%s)
for spec in "ATB_vs_OD GSE37250 withsex" "ATB_vs_OD GSE73408 withsex" "ATB_vs_OD GSE42834 withsex" "ATB_vs_OD GSE83456 withsex" \
            "ATB_vs_LTB GSE19491 nosex" "ATB_vs_LTB GSE37250 nosex" "ATB_vs_LTB GSE73408 nosex" "ATB_vs_LTB GSE101705 nosex" "ATB_vs_LTB GSE107994 nosex" \
            "ATB_vs_LTB GSE19491 withsex" "ATB_vs_LTB GSE37250 withsex" "ATB_vs_LTB GSE73408 withsex" "ATB_vs_LTB GSE101705 withsex" "ATB_vs_LTB GSE107994 withsex"; do
  set -- $spec
  [ -f "results/native/$1__fold_$2__$3.json" ] && continue
  LEFT=$(( 165 - ($(date +%s) - START) )); [ $LEFT -lt 40 ] && { echo "stop: time"; exit 0; }
  BUDGET=$(( LEFT - 25 )) timeout $LEFT python3 scripts/01_native.py fold $1 $2 $3 2>&1 | grep -v -E "stddev|RuntimeWarning" | tail -1
done
echo "queue complete"
