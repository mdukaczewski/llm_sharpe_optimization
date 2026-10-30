#!/usr/bin/env python3
"""NBBO check of the 15:59:30 cross entry used by the onml4 books: the books model a buy at the 15:59-bar close x (1 + predicted
hs1559) and a sell at x (1 - hs1559); here the real fill is the measured 15:59:30 ask / bid (out/quotes_panel.csv, ranks <= 600).
Also: official close vs the 15:59:30 mid (what an MOC fill gets relative to the continuous market)."""
import pandas as pd, numpy as np, sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization"); import rlib, liq
Q = pd.read_csv("out/quotes_panel.csv", dtype={"t": str}, parse_dates=["date"]); Q = Q[(Q.t == "1559")].dropna(subset=["bid", "ask"])
Q["rank"] = Q.tk.map(rlib.RANK); Q = Q[Q["rank"] <= 600]
rows = []
for tk, g in Q.groupby("tk"):
    try: M = rlib.Min(tk, end="2026-10-28"); L = liq.liq_frame(tk)
    except Exception: continue
    di = {d: i for i, d in enumerate(pd.DatetimeIndex(M.dates))}
    for _, r in g.iterrows():
        i = di.get(r.date)
        if i is None: continue
        rows.append((tk, r["rank"], r.date, r.bid, r.ask, M.c[i, 389], M.close[i], L.hs1559.get(r.date, np.nan)))
D = pd.DataFrame(rows, columns=["tk", "rank", "date", "bid", "ask", "c1559", "close", "hs_pred"])
D["mid"] = (D.bid + D.ask) / 2; D["hs_obs"] = (D.ask - D.bid) / (D.ask + D.bid) * 1e4
D["bar_vs_mid"] = (D.c1559 / D.mid - 1) * 1e4; D["close_vs_mid"] = (D.close / D.mid - 1) * 1e4
D["buy_err"] = (D.ask / (D.c1559 * (1 + D.hs_pred / 1e4)) - 1) * 1e4; D["sell_err"] = -(D.bid / (D.c1559 * (1 - D.hs_pred / 1e4)) - 1) * 1e4
D["tier"] = pd.cut(D["rank"], [0, 300, 600], labels=["1-300", "301-600"])
f = lambda x: pd.Series({"n": len(x), "hs_obs_med": x.hs_obs.median(), "hs_obs_mean": x.hs_obs.mean(), "hs_pred_mean": x.hs_pred.mean(),
    "absbar_mid_med": x.bar_vs_mid.abs().median(), "buy_err_mean": x.buy_err.mean(), "sell_err_mean": x.sell_err.mean(),
    "rt_err_mean": (x.buy_err + x.sell_err).mean() / 2, "close_vs_mid_mean": x.close_vs_mid.mean(), "absclose_mid_med": x.close_vs_mid.abs().median()})
pd.set_option("display.width", 220); print(D.groupby("tier", observed=True).apply(f).round(2).to_string())
print(D.assign(y=D.date.dt.year).groupby("y").apply(f)[["n", "hs_obs_mean", "hs_pred_mean", "rt_err_mean"]].round(2).to_string())
D.to_csv("out/cross1559_check.csv", index=False); print("CROSS DONE")
