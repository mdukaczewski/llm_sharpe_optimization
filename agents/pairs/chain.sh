#!/bin/bash
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
D=agents/pairs
while pgrep -f "agents/pairs/resid_rev.py" > /dev/null; do sleep 20; done
$PY -u $D/resid_rev3.py > $D/resid_rev3.log 2>&1
$PY -u $D/pairs.py > $D/pairs.log 2>&1
$PY -u $D/etf_spread.py > $D/etf_spread.log 2>&1
