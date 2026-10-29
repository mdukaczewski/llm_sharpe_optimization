"""Leak/microstructure diagnostic for the 10:30-exit intraday book: same model, books by liquidity tier,
delayed fill (decide 09:31, fill at the 09:33 bar = bin 3, exit bin 61), and harsher costs."""
import sys, time, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml"); sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import dlib, rlib, numpy as np, pandas as pd
from rlib import Min
t0 = time.time(); tks = rlib.universe(1500)
D = dlib.build(tks, verbose=False); print("rows", len(D), f"{time.time()-t0:.0f}s", flush=True)
# delayed-fill targets
ext = []
for tk in tks:
    M = Min(tk, end=rlib.DEV_END); c = M.c
    e = pd.DataFrame({"date": M.dates, "tk": tk, "y_60d": c[:, 61] / c[:, 3] - 1, "y_60vw": c[:, 60] / np.nanmean(c[:, 2:6], axis=1) - 1,
                      "nb": M.nbars, "spread_proxy": np.nanmedian((M.h_raw[:, 1:31] - M.l_raw[:, 1:31]) / c[:, 1:31], axis=1)})
    ext.append(e)
E = pd.concat(ext); D = D.merge(E, on=["date", "tk"], how="left")
for k in ["y_60d", "y_60vw"]:
    x = D[k].values.astype(float); x[np.abs(x) > 0.5] = np.nan; D[k] = x
print("extras merged", f"{time.time()-t0:.0f}s", flush=True)
Dp, pred = dlib.run(D, dlib.EXT, "y_60", retrain="Y", n_jobs=2, verbose=False)
print("IC", rlib.ic(Dp, pred, "y_60"), f"{time.time()-t0:.0f}s", flush=True)
pd.DataFrame({"date": Dp.date, "tk": Dp.tk, "pred": pred}).dropna().to_parquet("/home/user/llm_sharpe_optimization/agents/dayml/out/pred_y60_Y_1500.parquet")
rlib.book(Dp, pred, k=20, y="y_60", weight="pred", name="ALL  y_60 k20p (yearly)")
rlib.book(Dp, pred, k=20, y="y_60d", weight="pred", name="ALL  delayed fill bin3->bin61")
rlib.book(Dp, pred, k=20, y="y_60vw", weight="pred", name="ALL  fill avg bins2-5 -> bin60")
rlib.book(Dp, pred, k=20, y="y_60", weight="pred", cost=5.0, name="ALL  y_60 @5bps")
for lo, hi in [(1, 300), (301, 600), (601, 1000), (1001, 1500)]:
    m = (Dp["rank"] >= lo) & (Dp["rank"] <= hi); p2 = pred.where(m)
    rlib.book(Dp, p2, k=20, y="y_60", weight="pred", name=f"ranks {lo}-{hi} y_60 k20p")
    rlib.book(Dp, p2, k=20, y="y_60d", weight="pred", name=f"ranks {lo}-{hi} delayed fill")
# where do the book's positions sit?
g = Dp.loc[pred.notna(), ["date", "rank", "spread_proxy"]].copy(); g["p"] = pred[g.index].values
g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size")
sel = g[(g.rk <= 20) | (g.rk > g.n - 20)]
print("positions by rank tier:", np.histogram(sel["rank"], bins=[0, 300, 600, 1000, 1500])[0] / len(sel))
print("median intrabar range proxy (first 30 min) of selected names: %.4f  vs all: %.4f" % (sel.spread_proxy.median(), g.spread_proxy.median()))
print("DIAG DONE", flush=True)
