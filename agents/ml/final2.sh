#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
L=agents/ml/logs
rm -f $L/net_*.csv
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid final2 --smooth 5 --wsave final2_k25h5b0H_sm5 > $L/final2_sm5.log 2>&1
$PY -u agents/ml/ml_diag.py --w final2_k25h5b0H_sm5 > $L/diag_final2_k25h5b0H_sm5.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid final --wsave final2_k30h10b10H > $L/final2_sm1.log 2>&1
$PY -u agents/ml/ml_diag.py --w final2_k30h10b10H > $L/diag_final2_k30h10b10H.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --grid buf > $L/v3_ens3_buf.log 2>&1
echo FINAL2 DONE > $L/final2.done
