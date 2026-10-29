#!/bin/bash
cd /home/user/llm_sharpe_optimization
until grep -q "MINUTE DONE" logs/fetch_minute_local.log; do sleep 30; done
echo "minute done $(date)"
until grep -q "WARM DONE" agents/dayml/warm.log; do sleep 30; done
echo "warm done $(date)"
python3 agents/dayml/exp2.py y_120,y_60 Q
python3 agents/dayml/exp2.py y_day Y
echo "DRIVE2 DONE $(date)"
