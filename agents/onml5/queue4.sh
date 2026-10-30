cd /home/user/llm_sharpe_optimization/agents/onml5
until grep -q "QUEUE3 DONE" queue3.log; do sleep 30; done
python3 train5.py name=q_f26n feats=f26n rt=Q seeds=0,1,2 > train_q_f26n.log 2>&1
echo QUEUE4 DONE
