cd /home/user/llm_sharpe_optimization/agents/onml5
until grep -qE "DONE|Traceback" train_y_f40n_r600.log 2>/dev/null; do sleep 30; done
python3 eval5.py y_f40n_r600 y_f40n_r600+o3:y_f26x_on tag=y_f40n_r600 > eval_y_f40n_r600.log 2>&1
until grep -qE "DONE|Traceback" train_y_f40n_l600.log 2>/dev/null; do sleep 30; done
python3 eval5.py splice:y_f40n:y_f40n_l600 splice:y_f40n:y_f40n_l600+o3:y_f26x_on tag=l600 > eval_l600.log 2>&1
until grep -qE "DONE|Traceback" train_y_f26n.log 2>/dev/null; do sleep 30; done
python3 eval5.py y_f26n y_f26n+y_f40n y_f26n+y_f40 tag=y_f26n > eval_y_f26n.log 2>&1
echo EVALQ2 DONE
