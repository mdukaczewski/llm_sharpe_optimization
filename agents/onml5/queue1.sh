cd /home/user/llm_sharpe_optimization/agents/onml5
python3 train5.py name=y_f40 feats=f40 > train_y_f40.log 2>&1
python3 train5.py name=y_f40n_sqdv feats=f40n w=sqdv > train_y_f40n_sqdv.log 2>&1
python3 train5.py name=y_f40n_icost feats=f40n w=icost > train_y_f40n_icost.log 2>&1
python3 train5.py name=y_f40n_r600 feats=f40n tgt=r600 > train_y_f40n_r600.log 2>&1
python3 train5.py name=y_f40n_l600 feats=f40n filt=l600 > train_y_f40n_l600.log 2>&1
echo QUEUE1 DONE
