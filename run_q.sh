#!/bin/bash
cd /home/trumpet/strat20; export PYTHONPATH=/home/trumpet/strat20; PY=/home/trumpet/anaconda3/envs/py38/bin/python
S50=NVDA,MSFT,AMZN,AAPL,TSLA,META,GOOGL,AMD,AVGO,JPM,NFLX,COST,HD,WMT,UNH,V,LLY,XOM,BAC,ORCL,CRM,ADBE,CSCO,INTC,MU,QCOM,PLTR,UBER,COIN,PYPL,DIS,BA,GS,WFC,CAT,MCD,KO,PEP,ABBV,TMO,NKE,SBUX,JNJ,PG,CVX,PFE,MRK,TXN,C,SQ
A600=$S50,$(cat data/next100.txt),$(cat data/next150b.txt),$(cat data/next300c.txt)
echo "=== on_ml a600q (quarterly retrain)" >> logs/run_q.log
$PY -u on_ml.py $A600 1.0 a600q retrain=Q 2>&1 | grep -E "rows|IC|top|Error" >> logs/run_q.log
until grep -q "RUN1200 DONE" logs/run1200.log 2>/dev/null; do sleep 60; done
A1200=$A600,$(cat data/next600d.txt)
echo "=== on_ml a1200q (quarterly retrain)" >> logs/run_q.log
$PY -u on_ml.py $A1200 1.0 a1200q retrain=Q 2>&1 | grep -E "rows|IC|top|Error" >> logs/run_q.log
echo "=== on_ml a1200qk40" >> logs/run_q.log
$PY -u on_ml.py $A1200 1.0 a1200qk40 retrain=Q k=40 2>&1 | grep -E "IC|top|Error" >> logs/run_q.log
echo "RUNQ DONE" >> logs/run_q.log
