#!/bin/bash
cd /home/trumpet/strat20; export PYTHONPATH=/home/trumpet/strat20; PY=/home/trumpet/anaconda3/envs/py38/bin/python
S50=NVDA,MSFT,AMZN,AAPL,TSLA,META,GOOGL,AMD,AVGO,JPM,NFLX,COST,HD,WMT,UNH,V,LLY,XOM,BAC,ORCL,CRM,ADBE,CSCO,INTC,MU,QCOM,PLTR,UBER,COIN,PYPL,DIS,BA,GS,WFC,CAT,MCD,KO,PEP,ABBV,TMO,NKE,SBUX,JNJ,PG,CVX,PFE,MRK,TXN,C,SQ
ALL=$S50,$(cat data/next100.txt),$(cat data/next150b.txt)
echo "=== on_ml2 y300 (yearly)" >> logs/run_ml2.log
$PY -u on_ml2.py $ALL 1.0 y300 retrain=Y 2>&1 | grep -E "rows|IC|top|Error|error" >> logs/run_ml2.log
echo "=== on_ml2 q300 (quarterly)" >> logs/run_ml2.log
$PY -u on_ml2.py $ALL 1.0 q300 retrain=Q 2>&1 | grep -E "rows|IC|top|Error|error" >> logs/run_ml2.log
echo "RUNML2 DONE" >> logs/run_ml2.log
