#!/bin/bash
cd /home/trumpet/strat20; export PYTHONPATH=/home/trumpet/strat20; PY=/home/trumpet/anaconda3/envs/py38/bin/python
for t in $(cat data/next150b.txt | tr "," " "); do [ -f data/minute_npz/$t.npz ] || $PY agents/intraday/convert.py $t >> logs/run300.log 2>&1; done
S50=NVDA,MSFT,AMZN,AAPL,TSLA,META,GOOGL,AMD,AVGO,JPM,NFLX,COST,HD,WMT,UNH,V,LLY,XOM,BAC,ORCL,CRM,ADBE,CSCO,INTC,MU,QCOM,PLTR,UBER,COIN,PYPL,DIS,BA,GS,WFC,CAT,MCD,KO,PEP,ABBV,TMO,NKE,SBUX,JNJ,PG,CVX,PFE,MRK,TXN,C,SQ
N100=$(cat data/next100.txt); N150=$(cat data/next150b.txt)
echo "=== xs_on new150 smin=1" >> logs/run300.log
$PY -u xs_on.py $N150 1.0 new150s1 smin=1.0 2>&1 | grep -E "names|XSLO" >> logs/run300.log
echo "=== xs_on all300 smin=1" >> logs/run300.log
$PY -u xs_on.py $S50,$N100,$N150 1.0 all300s1 smin=1.0 2>&1 | grep -E "names|XSLO|TS long" >> logs/run300.log
echo "=== on_ml all300 exit1" >> logs/run300.log
$PY -u on_ml.py $S50,$N100,$N150 1.0 a300 2>&1 | grep -E "rows|IC|top" >> logs/run300.log
echo "=== on_ml all300 exit60" >> logs/run300.log
$PY -u on_ml.py $S50,$N100,$N150 1.0 a300x60 exit_bin=60 2>&1 | grep -E "IC|rule LO|lgb LS" >> logs/run300.log
echo "=== on_ml_day all300" >> logs/run300.log
$PY -u on_ml_day.py $S50,$N100,$N150 1.0 a300 2>&1 | grep -E "IC|lgb" >> logs/run300.log
echo "RUN300 DONE" >> logs/run300.log
