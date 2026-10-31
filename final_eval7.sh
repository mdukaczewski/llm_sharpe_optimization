#!/bin/bash
# Phase 7 (2026-10-31): every order uses only information available when it is sent.
#   overnight stock books: onml6 (signal cut at 15:49:00, MOC entry; or limit-on-close at the estimated 15:49 bid, fill iff the
#   official close is strictly through the limit), MOO exit, auction impact c=0.3; xsrule (15:59 signal into the MOC) dropped;
#   etfmacro `_ls` (15:59 SPY signal into the MOC) omitted; daily ML = dml2 (MOC on t-1 information, commission inside).
#   $3 per filled order on $0.4M per book, inverse-vol weights.      bash final_eval7.sh [--show_holdout]
cd "$(dirname "$0")"
OMIT="XSLOR|XSLOH|ens[34]eq|contango|backwd|mom_hedged|top10|top5$|XSLO5|XSLO10|XSLO15|XSTOP|XSLS|TS$|TSH|ridge|_x15|_x30|thr10|thr0_|shortvixy|S126|S252|E2_|top15|mom_.*top40|k40|ml_k30|h1_|_1545|_x1_|_ls$"
DROP="intraday,rule_e1,rule_late,lgb_on,dayml,ml,xsrule"
C="--sel all --wmode iv --holdout 2026-04-28 --omit $OMIT --drop $DROP --capital_usd 400000 --order_usd 3"
echo "=================== A: feasible timing (onml6 MOC + onml6 LOC as two sleeves)"
python3 folds.py $C --lanes onml=onml6moc,onmlloc=onml6loc,etfmacro=etfmacro,dml=dml2 "$@"
echo "=================== B: A + passive-limit intraday book (lmtday; fragile: second half <= 0, proxy fills)"
python3 folds.py $C --lanes onml=onml6moc,onmlloc=onml6loc,etfmacro=etfmacro,dml=dml2,lmtday=lmtday "$@"
echo "=================== C: reference, the previous headline with the infeasible 15:59 signal into the MOC (onml4x, xsrule3)"
python3 folds.py --sel all --wmode iv --holdout 2026-04-28 --omit "${OMIT%|_ls\$}" --drop intraday,rule_e1,rule_late,lgb_on,dayml,ml --capital_usd 400000 --order_usd 3 --lanes onml=onml4x,xsrule=xsrule3,etfmacro=etfmacro,dml=dml2 "$@"
