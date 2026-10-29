#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
L=agents/ml/logs
rm -f agents/ml/cache/feats_50.npz
for t in "lgb5n --h 5" "lgb1n --h 1" "lgb10n --h 10" "lgb5r --h 5 --target res" "lgb1r --h 1 --target res" "lgb10r --h 10 --target res"; do
  set -- $t; tag=$1; shift
  $PY -u agents/ml/ml_run.py --tag $tag "$@" --nomk > $L/${tag}_v2.log 2>&1
done
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid main > $L/v2_ens3_main.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid buf > $L/v2_ens3_buf.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1r,lgb5r,lgb10r --grid buf > $L/v2_ens3r_buf.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n,lgb1r,lgb5r,lgb10r --grid buf > $L/v2_ens6_buf.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid buf --short_adv 200e6 --short_px 10 > $L/v2_ens3_buf_shortliq.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid main --smooth 5 > $L/v2_ens3_sm5.log 2>&1
echo QUEUE4 DONE > $L/queue4.done
