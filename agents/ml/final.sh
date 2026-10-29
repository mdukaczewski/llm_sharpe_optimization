#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
L=agents/ml/logs
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid final --wsave final_k30h10b10H > $L/final_sm1.log 2>&1
$PY -u agents/ml/ml_diag.py --w final_k30h10b10H > $L/diag_final.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid final --smooth 5 > $L/final_sm5.log 2>&1
echo FINAL DONE > $L/final.done
