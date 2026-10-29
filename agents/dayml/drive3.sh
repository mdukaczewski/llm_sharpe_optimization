#!/bin/bash
cd /home/user/llm_sharpe_optimization
python3 agents/dayml/exp2.py y_120,y_60 Q
python3 agents/dayml/exp2.py y_day Y
echo "DRIVE3 DONE $(date)"
