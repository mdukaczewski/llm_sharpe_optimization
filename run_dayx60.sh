#!/bin/bash
cd /home/trumpet/strat20; export PYTHONPATH=/home/trumpet/strat20; PY=/home/trumpet/anaconda3/envs/py38/bin/python
until grep -q 'RUNML2X60 DONE' logs/run_ml2.log; do sleep 30; done
S50=NVDA,MSFT,AMZN,AAPL,TSLA,META,GOOGL,AMD,AVGO,JPM,NFLX,COST,HD,WMT,UNH,V,LLY,XOM,BAC,ORCL,CRM,ADBE,CSCO,INTC,MU,QCOM,PLTR,UBER,COIN,PYPL,DIS,BA,GS,WFC,CAT,MCD,KO,PEP,ABBV,TMO,NKE,SBUX,JNJ,PG,CVX,PFE,MRK,TXN,C,SQ
ALL=$S50,$(cat data/next100.txt),$(cat data/next150b.txt)
echo '=== on_ml_day a300x60 (09:32 -> 10:31)' >> logs/run_ml2.log
$PY -u on_ml_day.py $ALL 1.0 a300x60 exit_bin=61 2>&1 | grep -E 'rows|IC|top|Error|error' >> logs/run_ml2.log
echo 'RUNDAYX60 DONE' >> logs/run_ml2.log
