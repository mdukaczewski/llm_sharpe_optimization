#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
L=agents/ml/logs
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid iv > $L/v2_ens3_iv.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb5o --h 5 --target oc --nomk > $L/lgb5o_v2.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb1o --h 1 --target oc --nomk > $L/lgb1o_v2.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb10o --h 10 --target oc --nomk > $L/lgb10o_v2.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1o,lgb5o,lgb10o --grid buf > $L/v2_ens3o_buf.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n,lgb1o,lgb5o,lgb10o --grid buf > $L/v2_ens6o_buf.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb5big --h 5 --nomk --leaves 63 --rounds 800 --lr 0.02 > $L/lgb5big_v2.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb5big --grid buf > $L/v2_lgb5big_buf.log 2>&1
echo QUEUE5 DONE > $L/queue5.done
