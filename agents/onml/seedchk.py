import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
tks = rlib.universe(1500)[:143]
D = rlib.minute_frame(tks, verbose=False); D = rlib.add_market(D); D["yr"] = D.groupby("date").y_on.rank(pct=True) - 0.5; D = rlib.clean(D, FEATS18)
p = rlib.walk_forward(D, FEATS18, "yr", retrain="Y", n_jobs=2, verbose=False)
r = rlib.book(D, p, k=20, p=False); log(report_line("exact baseline order/default seed", r))
D2 = D.sort_values(["tk", "date"]).reset_index(drop=True)
rs = {}
for s in [0, 1, 2]:
    p = wf(D2, FEATS18, "yr", "Y", seed=s); rs[s] = rlib.book(D2, p, k=20, p=False); log(report_line(f"sorted seed{s}", rs[s]))
log(report_line("seed-avg returns 0,1,2", sum(rs.values()) / 3))
