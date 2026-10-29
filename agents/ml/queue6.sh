#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
L=agents/ml/logs
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --topn_univ 500 --grid main > $L/v2_ens3_top500_main.log 2>&1
$PY -u agents/ml/ml_port.py --pred lgb1n,lgb5n,lgb10n --topn_univ 500 --grid buf > $L/v2_ens3_top500_buf.log 2>&1
$PY -u agents/ml/ml_run.py --tag s1 --h 1 --min_adv 10e6 --nomk --step 2 > $L/s1_v2.log 2>&1
$PY -u agents/ml/ml_run.py --tag s5 --h 5 --min_adv 10e6 --nomk --step 2 > $L/s5_v2.log 2>&1
$PY -u agents/ml/ml_port.py --pred s1,s5 --min_adv 10e6 --cost 20 --grid main > $L/v2_s_main.log 2>&1
$PY -u agents/ml/ml_port.py --pred s1,s5 --min_adv 10e6 --cost 20 --grid buf > $L/v2_s_buf.log 2>&1
$PY -u agents/ml/ml_port.py --pred s1,s5 --min_adv 10e6 --cost 20 --grid buf --short_adv 50e6 --short_px 10 > $L/v2_s_buf_shortliq.log 2>&1
echo QUEUE6 DONE > $L/queue6.done
