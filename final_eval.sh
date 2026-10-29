#!/bin/bash
# Final fold validation (dev folds 2018-07-01..2026-04-27, picks from prior dates only) and the single holdout
# look (2026-04-28..2026-10-28) for the 2026-10-29 session.  Families: the original coarse grid (README) minus the
# old 09:32-fill intraday family (bid-ask bounce, see agents/dayml/diag_liq.py), plus the four agent lanes
# (agents/<lane>/cands).  Reported at three cost levels: spread-only (1-1.5 bps/side), +0.6 bps/side commission
# $3 per order on $0.4M equity per book (user's figure; L/S top-20 = 40 positions of $20k = 1.5 bps/side) and on $2M per book.
#   bash final_eval.sh [--show_holdout]
cd "$(dirname "$0")"
OMIT="XSLOR|XSLOH|ens[34]eq|contango|backwd|mom_hedged|top10|top5$|XSLO5|XSLO10|XSLO15|XSTOP|XSLS|TS$|TSH|ridge|_x15|_x30|thr10|thr0_|shortvixy|S126|S252|E2_|top15|mom_.*top40|k40|ml_k30|h1_|pred_lgbv|pred_lgbbig|pred_ens2|onml2|a300x60|c15|cBx|_1545|onml_f28cal_q_k20"
echo "=================== spread only (1-1.5 bps/side), no commission"
python3 folds.py --sel all --wmode iv --holdout 2026-04-28 --omit "$OMIT" --drop intraday "$@"
for C in 400000 2000000; do
  echo "=================== \$3 per order, \$${C} equity per book"
  python3 folds.py --sel all --wmode iv --holdout 2026-04-28 --omit "$OMIT" --drop intraday --capital_usd $C --order_usd 3 "$@"
done
