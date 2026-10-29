#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
L=agents/ml/logs
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid main > $L/port_ens3.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid legs > $L/port_ens3_legs.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid fine > $L/port_ens3_fine.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb5r --h 5 --target res --nomk > $L/lgb5r.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb1r --h 1 --target res --nomk > $L/lgb1r.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb10r --h 10 --target res --nomk > $L/lgb10r.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1r,lgb5r,lgb10r --grid main > $L/port_ens3r.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb5w --h 5 --window 756 --nomk > $L/lgb5w.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb5w --grid main > $L/port_lgb5w.log 2>&1
echo QUEUE2 DONE > $L/queue2.done
