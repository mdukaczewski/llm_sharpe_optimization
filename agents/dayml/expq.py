"""quarterly vs yearly retrain, y_120 EXT, on the saved 269-name frame"""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml"); import dlib, rlib, pandas as pd
OUT = "/home/user/llm_sharpe_optimization/agents/dayml/out"; D = pd.read_parquet(f"{OUT}/D_400.parquet")
t = time.time(); Dp, pred = dlib.run(D, dlib.EXT, "y_120", retrain="Q"); pred.rename("q").to_frame().to_parquet(f"{OUT}/pred_ext_y120Q_400.parquet")
print(f"--- ext_y120 QUARTERLY {time.time()-t:.0f}s IC {rlib.ic(Dp, pred, 'y_120')}", flush=True)
res = {}
for k in (10, 20, 30): res[f"q_k{k}"] = dlib.book(Dp, pred, k=k, y="y_120", weight="pred", name=f"ext_y120 Q k{k} pred")
for c in res: rlib.fold_oos(res[c], label=c)
print("DONE expq", flush=True)
