#!/bin/bash
cd /home/trumpet/strat20; export PYTHONPATH=/home/trumpet/strat20; PY=/home/trumpet/anaconda3/envs/py38/bin/python
for t in $(cat data/next300c.txt | tr "," " "); do [ -f data/minute_npz/$t.npz ] || $PY agents/intraday/convert.py $t >> logs/run600.log 2>&1; done
S50=NVDA,MSFT,AMZN,AAPL,TSLA,META,GOOGL,AMD,AVGO,JPM,NFLX,COST,HD,WMT,UNH,V,LLY,XOM,BAC,ORCL,CRM,ADBE,CSCO,INTC,MU,QCOM,PLTR,UBER,COIN,PYPL,DIS,BA,GS,WFC,CAT,MCD,KO,PEP,ABBV,TMO,NKE,SBUX,JNJ,PG,CVX,PFE,MRK,TXN,C,SQ
A300=$S50,$(cat data/next100.txt),$(cat data/next150b.txt); N300=$(cat data/next300c.txt); A600=$A300,$N300
echo "=== xs_on new300 smin=1" >> logs/run600.log
$PY -u xs_on.py $N300 1.0 new300s1 smin=1.0 2>&1 | grep -E "names|XSLO" >> logs/run600.log
echo "=== xs_on all600 smin=1" >> logs/run600.log
$PY -u xs_on.py $A600 1.0 all600s1 smin=1.0 2>&1 | grep -E "names|XSLO|TS long" >> logs/run600.log
echo "=== on_ml all600 exit1" >> logs/run600.log
$PY -u on_ml.py $A600 1.0 a600 2>&1 | grep -E "rows|IC|top|Error" >> logs/run600.log
echo "=== on_ml all600 exit1 k40" >> logs/run600.log
$PY -u on_ml.py $A600 1.0 a600k40 k=40 2>&1 | grep -E "IC|top|Error" >> logs/run600.log
echo "=== on_ml all600 exit60" >> logs/run600.log
$PY -u on_ml.py $A600 1.0 a600x60 exit_bin=60 2>&1 | grep -E "IC|rule LO|lgb LS|Error" >> logs/run600.log
echo "=== on_ml_day all600" >> logs/run600.log
$PY -u on_ml_day.py $A600 1.0 a600 2>&1 | grep -E "IC|lgb|Error" >> logs/run600.log
echo "RUN600 DONE" >> logs/run600.log
