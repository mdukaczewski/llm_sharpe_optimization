import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import rlib, numpy as np, pandas as pd
t0 = time.time()
tks = rlib.universe(150); print(len(tks), "names", flush=True)
D = rlib.add_market(rlib.minute_frame(tks)); D = rlib.clean(D, rlib.FEATS18)
print("rows", len(D), f"{time.time()-t0:.0f}s", flush=True)
D["yr"] = D.groupby("date").y_on.rank(pct=True) - 0.5
pred = rlib.walk_forward(D, rlib.FEATS18, "yr", retrain="Y", n_jobs=3, verbose=False)
print("IC", rlib.ic(D, pred), f"{time.time()-t0:.0f}s", flush=True)
rlib.book(D, pred, k=20, mode="ls", name="lgb LS top20 (150 names, yearly)")
rlib.book(D, pred, k=20, mode="lo", name="lgb LO top20")
rule = 0.5 * D.z30 - D.z60
rlib.book(D, rule, k=20, mode="lo", smin=1.0, name="rule XSLO20 s>1")
ref = pd.read_csv("sleeves/onlgb_ls20.csv", index_col=0, parse_dates=True).iloc[:, 0]
rlib.stats(ref, "README sleeve onlgb_ls20 (150 names)")
ref = pd.read_csv("sleeves/xslo20s1_150.csv", index_col=0, parse_dates=True).iloc[:, 0]
rlib.stats(ref, "README sleeve xslo20s1_150")
