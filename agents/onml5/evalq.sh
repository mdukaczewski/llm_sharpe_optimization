cd /home/user/llm_sharpe_optimization/agents/onml5
for n in y_f40n_sqdv y_f40n_icost y_f40n_r600 y_f26n y_f40a y_f40c; do
  until grep -qE "DONE|Traceback" train_$n.log 2>/dev/null; do sleep 30; done
  python3 eval5.py $n $n+o3:y_f26x_on tag=$n > eval_$n.log 2>&1
done
until grep -qE "DONE|Traceback" train_y_f40n_l600.log 2>/dev/null; do sleep 30; done
python3 eval5.py splice:y_f40n:y_f40n_l600 splice:y_f40n:y_f40n_l600+o3:y_f26x_on y_f26n+y_f40n tag=l600 > eval_l600.log 2>&1
echo EVALQ DONE
