#!/bin/bash
# waits for the minute-bar fetchers to finish, converts, then runs the 1200-name books
cd /home/trumpet/strat20; export PYTHONPATH=/home/trumpet/strat20; PY=/home/trumpet/anaconda3/envs/py38/bin/python
until ! pgrep -f "[f]etch_minute.py" > /dev/null; do sleep 60; done
sleep 120
for t in $(cat data/next600d.txt | tr "," " "); do [ -f data/minute_npz/$t.npz ] || $PY agents/intraday/convert.py $t >> logs/run1200.log 2>&1; done
S50=NVDA,MSFT,AMZN,AAPL,TSLA,META,GOOGL,AMD,AVGO,JPM,NFLX,COST,HD,WMT,UNH,V,LLY,XOM,BAC,ORCL,CRM,ADBE,CSCO,INTC,MU,QCOM,PLTR,UBER,COIN,PYPL,DIS,BA,GS,WFC,CAT,MCD,KO,PEP,ABBV,TMO,NKE,SBUX,JNJ,PG,CVX,PFE,MRK,TXN,C,SQ
A600=$S50,$(cat data/next100.txt),$(cat data/next150b.txt),$(cat data/next300c.txt); N600=$(cat data/next600d.txt); A1200=$A600,$N600
echo "=== xs_on new600 smin=1" >> logs/run1200.log
$PY -u xs_on.py $N600 1.0 new600s1 smin=1.0 2>&1 | grep -E "names|XSLO" >> logs/run1200.log
echo "=== xs_on all1200 smin=1" >> logs/run1200.log
$PY -u xs_on.py $A1200 1.0 all1200s1 smin=1.0 2>&1 | grep -E "names|XSLO|TS long" >> logs/run1200.log
echo "=== on_ml all1200 exit1" >> logs/run1200.log
$PY -u on_ml.py $A1200 1.0 a1200 2>&1 | grep -E "rows|IC|top|Error" >> logs/run1200.log
echo "=== on_ml all1200 exit1 k40" >> logs/run1200.log
$PY -u on_ml.py $A1200 1.0 a1200k40 k=40 2>&1 | grep -E "IC|top|Error" >> logs/run1200.log
echo "=== on_ml all1200 exit1 1.5bps" >> logs/run1200.log
$PY -u on_ml.py $A1200 1.5 a1200c15 2>&1 | grep -E "lgb LS|Error" >> logs/run1200.log
echo "=== on_ml_day all1200" >> logs/run1200.log
$PY -u on_ml_day.py $A1200 1.0 a1200 2>&1 | grep -E "IC|lgb|Error" >> logs/run1200.log
echo "=== on_ml all1200 exit60" >> logs/run1200.log
$PY -u on_ml.py $A1200 1.0 a1200x60 exit_bin=60 2>&1 | grep -E "IC|rule LO|lgb LS|Error" >> logs/run1200.log
echo "RUN1200 DONE" >> logs/run1200.log
