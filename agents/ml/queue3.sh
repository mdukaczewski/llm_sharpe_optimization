#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
L=agents/ml/logs
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid best --wsave ens3_k40h5b20H > $L/port_ens3_best.log 2>&1
$PY -u agents/ml/ml_diag.py --w ens3_k40h5b20H > $L/diag_ens3.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid buf > $L/port_ens3_buf.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid main --smooth 5 > $L/port_ens3_sm5.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1r,lgb5r,lgb10r --grid buf > $L/port_ens3r_buf.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n,lgb1r,lgb5r,lgb10r --grid buf > $L/port_ens6_buf.log 2>&1
$PY -u agents/ml/ml_run.py --tag rank5n --h 5 --model rank --nomk > $L/rank5n.log 2>&1
$PY -u agents/ml/ml_port.py --pred rank5n --grid buf > $L/port_rank5n_buf.log 2>&1
echo QUEUE3 DONE > $L/queue3.done
