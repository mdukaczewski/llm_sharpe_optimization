#!/bin/bash
# Phase 6 (2026-10-30): realistic execution.  Minute-bar families come ONLY from the phase-6 lanes (series net of the modelled
# auction impact at c=0.3 or of the spread fraction named in the candidate, before commission); the old 1-2.5 bps/side
# minute-bar families (lgb_on, onml, dayml, xsrule, intraday, rule_*) are dropped.  Daily-bar sleeves (mom, vol, ml) and the
# VIX-ETP book keep their costs.  $3 per order on $0.4M per book, inverse-vol weights.
#   bash final_eval6.sh [--show_holdout]
cd "$(dirname "$0")"
OMIT="XSLOR|XSLOH|ens[34]eq|contango|backwd|mom_hedged|top10|top5$|XSLO5|XSLO10|XSLO15|XSTOP|XSLS|TS$|TSH|ridge|_x15|_x30|thr10|thr0_|shortvixy|S126|S252|E2_|top15|mom_.*top40|k40|ml_k30|h1_|_1545|_x1_"
LANES="onml=onml3,dayml=dayml3,xsrule=xsrule3,etfmacro=etfmacro"
DROP="intraday,rule_e1,rule_late,lgb_on"
for W in iv; do
  echo "=================== phase-6 lanes, \$3 per order, \$400000 per book, weights=$W (x1 = crossing-the-spread variants omitted)"
  python3 folds.py --sel all --wmode $W --holdout 2026-04-28 --omit "$OMIT" --drop $DROP --lanes $LANES --capital_usd 400000 --order_usd 3 "$@"
  echo "=================== same, crossing-the-spread variants included"
  python3 folds.py --sel all --wmode $W --holdout 2026-04-28 --omit "${OMIT%|_x1_}" --drop $DROP --lanes $LANES --capital_usd 400000 --order_usd 3 "$@"
done
