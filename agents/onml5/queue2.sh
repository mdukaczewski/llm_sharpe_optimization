cd /home/user/llm_sharpe_optimization/agents/onml5
until grep -q "QUEUE1 DONE" queue1.log; do sleep 30; done
python3 train5.py name=y_f26n feats=f26n > train_y_f26n.log 2>&1
python3 train5.py name=y_f40a feats=f40a > train_y_f40a.log 2>&1
python3 train5.py name=y_f40c feats=f40c > train_y_f40c.log 2>&1
echo QUEUE2 DONE
