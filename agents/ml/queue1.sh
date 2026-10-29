#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
L=agents/ml/logs
$PY -u agents/ml/ml_run.py --tag lgb5n --h 5 --nomk > $L/lgb5n.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb5n --grid main > $L/port_lgb5n.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb10n --h 10 --nomk > $L/lgb10n.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb10n --grid main > $L/port_lgb10n.log 2>&1
$PY -u agents/ml/ml_run.py --tag lgb1n --h 1 --nomk > $L/lgb1n.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n --grid main > $L/port_lgb1n.log 2>&1
$PY -u agents/ml/ml_run.py --tag ridge5n --h 5 --model ridge --nomk > $L/ridge5n.log 2>&1
$PY -u agents/ml/ml_port.py --pred ridge5n --grid main > $L/port_ridge5n.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb5 --grid main > $L/port_lgb5.log 2>&1
echo QUEUE1 DONE > $L/queue1.done
