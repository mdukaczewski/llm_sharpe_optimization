#!/bin/bash
cd /home/trumpet/strat20; export PYTHONPATH=/home/trumpet/strat20; PY=/home/trumpet/anaconda3/envs/py38/bin/python
S50=NVDA,MSFT,AMZN,AAPL,TSLA,META,GOOGL,AMD,AVGO,JPM,NFLX,COST,HD,WMT,UNH,V,LLY,XOM,BAC,ORCL,CRM,ADBE,CSCO,INTC,MU,QCOM,PLTR,UBER,COIN,PYPL,DIS,BA,GS,WFC,CAT,MCD,KO,PEP,ABBV,TMO,NKE,SBUX,JNJ,PG,CVX,PFE,MRK,TXN,C,SQ
N100=$(cat data/next100.txt); N150=$(cat data/next150b.txt)
$PY -u xs_on.py $S50,$N100 1.0 all150s1 smin=1.0 2>&1 | grep -E "XSLOH" > logs/run_xsh.log
$PY -u xs_on.py $S50,$N100,$N150 1.0 all300s1 smin=1.0 2>&1 | grep -E "XSLOH" >> logs/run_xsh.log
$PY -u xs_on.py $S50,$N100 1.0 all150 2>&1 | grep -E "XSLOH" >> logs/run_xsh.log
echo "RUNXSH DONE" >> logs/run_xsh.log
