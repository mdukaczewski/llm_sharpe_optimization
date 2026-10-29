import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
tks = rlib.universe(1500)[:143]
D = load(tickers=tks, extra=False)
p = wf(D, FEATS18, "yr", "Y"); log("IC", rlib.ic(D, p))
r = rlib.book(D, p, k=20, name="repro lgb LS top20 143 names yearly")
log(report_line("repro143_Y", r))
