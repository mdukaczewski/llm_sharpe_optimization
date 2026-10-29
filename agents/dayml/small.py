"""Save the small intraday books (k6/k8 eq, both exits) as candidates from the saved predictions (full range)."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml"); sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import dlib, rlib, pandas as pd
tks = rlib.universe(1500); D = dlib.build(tks, verbose=False)
for y, tag in [("y_60d", "x60d"), ("y_120d", "x120d")]:
    P = pd.read_parquet(f"/home/user/llm_sharpe_optimization/agents/dayml/out/pred_{y}_Q.parquet")
    M = D[["date", "tk", y, "rv20"]].merge(P, on=["date", "tk"], how="left")
    for k in [6, 8]:
        r = rlib.book(M, M["p"], k=k, y=y, mode="ls", weight="eq", name=f"dayml_{tag}q_k{k}")
        rlib.save_cand("dayml", f"dayml_{tag}q_k{k}", r, f"09:31 LightGBM quarterly, fill 09:33 bar, exit {'10:31' if y == 'y_60d' else '11:31'} bar, L/S top{k} eq-weight (commission regime)")
print("SMALL DONE")
