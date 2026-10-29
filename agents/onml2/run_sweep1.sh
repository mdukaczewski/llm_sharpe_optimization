#!/bin/bash
cd /home/user/llm_sharpe_optimization
while pgrep -f "preds.py" > /dev/null; do sleep 30; done
echo "$(date +%T) preds.py gone, starting sweep1" >> agents/onml2/log.txt
nice -n 5 python3 agents/onml2/sweep1.py >> agents/onml2/out/sweep1.log 2>&1
