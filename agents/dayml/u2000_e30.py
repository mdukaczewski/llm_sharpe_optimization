"""Apply the dayml2 construction (exit at bin 31 = 10:01, k8 eq) to the 2000-name y_60d predictions; full range per STRAT_END,
printed stats cut at DEV_CUT."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml2"); sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml"); sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import rlib, d2lib, numpy as np, pandas as pd
CAP = 400000.0; DEV_CUT = "2026-04-27"; t0 = time.time()
tks = rlib.universe(2000)
parts = []
for tk in tks:
    try: e = d2lib.extra2(tk)[["y_30d"]].reset_index(); e["tk"] = tk; parts.append(e)
    except Exception as ex: print(tk, "skip", ex)
E = pd.concat(parts); E["date"] = pd.to_datetime(E["date"]); print("targets", len(E), f"{time.time()-t0:.0f}s", flush=True)
P = pd.read_parquet("/home/user/llm_sharpe_optimization/agents/dayml/out/pred_y_60d_Q_u2000.parquet")
M = P.merge(E, on=["date", "tk"], how="left"); M["rv20"] = np.nan; M = M[M.date <= rlib.DEV_END]
for k in [6, 8, 10]:
    r = rlib.book(M, M["p"], k=k, y="y_30d", mode="ls", weight="eq", p=False); name = f"dayml_x60dq_u2000_e30_k{k}"
    rlib.save_cand("dayml", name, r, f"09:31 LightGBM quarterly (y_60d target) on universe(2000), fill 09:33 bar, EXIT 10:01 bar (bin 31), L/S top{k} eq-weight")
    rd = r[r.index <= DEV_CUT]; rn = rd - 2 * k * 6 / CAP
    print(f"{name:28s} spread-only Sh {rlib.sh(rd):.2f} foldOOS {rlib.sh(rlib.fold_oos(rd, p=False)[0]):.2f} | $0.4M Sh {rlib.sh(rn):.2f} foldOOS {rlib.sh(rlib.fold_oos(rn, p=False)[0]):.2f} | halves {rlib.sh(rn[rn.index < '2021-10-28']):.2f}/{rlib.sh(rn[rn.index >= '2021-10-28']):.2f}", flush=True)
print("E30 DONE", flush=True)
