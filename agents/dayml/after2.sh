#!/bin/bash
cd /home/user/llm_sharpe_optimization
until grep -q "DRIVE2 DONE" agents/dayml/log2.txt; do sleep 60; done
python3 agents/dayml/post2.py >> agents/dayml/log2.txt 2>&1
