cd /home/user/llm_sharpe_optimization/agents/onml5
until grep -q "QUEUE1 DONE" queue1.log; do sleep 30; done
python3 train5.py name=y_f26n feats=f26n > train_y_f26n.log 2>&1
python3 train5.py name=q_f40n feats=f40n rt=Q seeds=0,1,2 > train_q_f40n.log 2>&1
echo QUEUE3 DONE
