#!/usr/bin/env python3
"""Coordinator check: NBBO at 15:59:30 for the ACTUAL 15:59:30-cross entries of onml4_pR_lam3_rte_moo_k6 on the quote-panel days.
Model fill: c389 x (1 +- predicted hs1559); real fill: ask (buy) / bid (sell short) at 15:59:30."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml4")
from lib4 import *
G = load_G()
G["pR"] = (((G.p40 + G.p26) / 2).groupby(G.date).rank(pct=True) - 0.5).astype(np.float32)
H = prep(G, "pR"); r, L = route_book(H, 6, 3.0, "route", "moo", 0.3, 1.0, ret_pos=True)
Q = pd.read_csv("/home/user/llm_sharpe_optimization/out/quotes_panel.csv", dtype={"t": str}, parse_dates=["date"]); Q = Q[Q.t == "1559"].dropna(subset=["bid", "ask"])
P = L.merge(Q[["tk", "date", "bid", "ask"]], on=["tk", "date"], how="inner")
print("positions on panel days:", len(P), "of which 15:59:30 crosses:", (P.entry == "x1559").sum(), "MOC:", (P.entry == "moc").sum())
c389 = {}
for tk in P.tk.unique():
    M = rlib.Min(tk, end="2026-10-28"); s = pd.Series(M.c[:, 389], index=pd.DatetimeIndex(M.dates)); c389[tk] = s
P["c389"] = [c389[t].get(d, np.nan) for t, d in zip(P.tk, P.date)]
P = P.merge(G[["date", "tk", "hs1559"]], on=["date", "tk"], how="left")
long = P.w > 0
P["model_fill"] = np.where(long, P.c389 * (1 + P.hs1559 / 1e4), P.c389 * (1 - P.hs1559 / 1e4))
P["real_fill"] = np.where(long, P.ask, P.bid)
P["err_bps"] = np.where(long, P.real_fill / P.model_fill - 1, 1 - P.real_fill / P.model_fill) * 1e4     # + = real fill worse than modelled
X = P[P.entry == "x1559"]
for nm, g in [("all crosses", X), ("long", X[X.w > 0]), ("short", X[X.w < 0])]:
    print(f"{nm:12s} n {len(g):4d}  mean err {g.err_bps.mean():+.2f} bps (t {g.err_bps.mean()/g.err_bps.std()*np.sqrt(len(g)):+.1f})  median {g.err_bps.median():+.2f}  obs hs {((g.ask-g.bid)/(g.ask+g.bid)*1e4).mean():.2f} vs pred {g.hs1559.mean():.2f}")
print("book gross per position (bps, all days):", round(L.y.mean() * 1e4, 2), " -> error as a share of gross:", round(X.err_bps.mean() / (L.y.mean() * 1e4) * 100, 1), "%")
