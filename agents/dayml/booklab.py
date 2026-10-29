"""Commission-aware book lab over saved intraday predictions: P positions x $6 / $400k per day on top of spread."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml"); sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import dlib, rlib, numpy as np, pandas as pd
CAP = 400000.0; t0 = time.time()
tks = rlib.universe(1500); D = dlib.build(tks, verbose=False); D = D[D.date <= rlib.DEV_END]
print("rows", len(D), f"{time.time()-t0:.0f}s", flush=True)
rows = []
for y, tag in [("y_60d", "x60d"), ("y_120d", "x120d")]:
    P = pd.read_parquet(f"/home/user/llm_sharpe_optimization/agents/dayml/out/pred_{y}_Q.parquet")
    M = D[["date", "tk", y, "rv20", "n_cs"]].merge(P, on=["date", "tk"], how="left"); pred = M["p"]
    for mode in ["ls", "lo"]:
        for k in [6, 8, 10, 12, 15, 20]:
            for w in ["pred", "eq"]:
                r = rlib.book(M, pred, k=k, y=y, mode=mode, weight=w, p=False); npos = k * (2 if mode == "ls" else 1)
                rn = r - npos * 6 / CAP; fo = rlib.fold_oos(rn, p=False)[0]
                rows.append((tag, mode, k, w, "-", round(rlib.sh(r), 2), round(rlib.sh(rn), 2), round(rlib.sh(fo), 2), f"{rn.mean()*252*100:.0f}%"))
        # variable size: only names whose |p| (centred rank) is beyond a per-day quantile, capped at k
        for q in [0.97, 0.985, 0.99]:
            thr = pred.abs().groupby(M.date).transform(lambda s: s.quantile(q)); pq = pred.where(pred.abs() >= thr)
            for k in [10, 20]:
                r = rlib.book(M, pq, k=k, y=y, mode=mode, weight="pred", p=False)
                # positions per day = count of selected names (approx: names beyond quantile, capped 2k)
                npos_day = pq.notna().groupby(M.date).sum().clip(upper=k * (2 if mode == "ls" else 1)).reindex(r.index).fillna(0)
                rn = r - npos_day * 6 / CAP; fo = rlib.fold_oos(rn, p=False)[0]
                rows.append((tag, mode, k, "pred", f"q{q}", round(rlib.sh(r), 2), round(rlib.sh(rn), 2), round(rlib.sh(fo), 2), f"{rn.mean()*252*100:.0f}%"))
T = pd.DataFrame(rows, columns=["book", "mode", "k", "w", "thr", "Sh_spread", "Sh_$0.4M", "foldOOS_$0.4M", "ret_$0.4M"])
pd.set_option("display.width", 200); print(T.sort_values("foldOOS_$0.4M", ascending=False).to_string(index=False))
T.to_csv("/home/user/llm_sharpe_optimization/agents/dayml/booklab.csv", index=False); print("LAB DONE", flush=True)
