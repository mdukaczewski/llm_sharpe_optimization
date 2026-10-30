"""Exp 5: the short side of the x1 signal alone (short ETF i MOC->MOO when s_i < -th), as a diversifier to the long-overnight stock books."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, elib2 as E
from rlib import sh
H = os.path.dirname(os.path.abspath(__file__))
D = E.frame(E.ALL); D = D[~D.half.astype(bool)].copy()
D["s"] = 0.5 * D.z30 - D.z60; D["rvl"] = D.groupby("tk").rv20.shift(1)
Pv = lambda c: D.pivot(index="date", columns="tk", values=c)
S, RV, Y, Y1 = Pv("s"), Pv("rvl"), Pv("y_oo"), Pv("y_on")
def cap(W, g=2.0):
    s_ = W.abs().sum(1); return W.mul(np.minimum(1.0, g / s_.replace(0, np.nan)).fillna(1.0), axis=0).fillna(0)
X = E.refs()["xsr3_lo10"]; O = E.refs()["onml3_k6"]
res = {}
E.log("=== x5 short-only overnight books (short when s < -th), 0.05/rv20 per position, borrow 1%, $3/order inside")
for un, U in [("eq", E.EQ), ("all", E.ALL), ("idx4", ["SPY", "QQQ", "IWM", "DIA"])]:
    base = (0.05 / RV[U]).clip(upper=0.5)
    for th in [0.5, 1.0, 1.5]:
        W = cap(-base.where(S[U] < -th, 0.0))
        for cost, yy, tg in [(1.5, Y, "moo"), (3.0, Y, "moo"), (1.5, Y1, "b931")]:
            n, g, npos, gr = E.run_book(W, yy[U], cost); nm = f"{un}_short_s{th}_{tg}_c{cost}"; res[nm] = n
            E.report(nm, n, npos, gr)
        n = res[f"{un}_short_s{th}_moo_c1.5"]; a = pd.concat([n, X, O], axis=1, keys=["e", "x", "o"]).dropna()
        b = np.polyfit(a.x, a.e, 1)[0]; r_ = a.e - b * a.x
        cm = 0.75 * a.x / a.x.std() + 0.25 * a.e / a.e.std(); cm0 = a.x / a.x.std()
        E.log(f"    -> vs xsr3: beta {b:+.2f} resid Sh {sh(r_):.2f}; 0.75 xsr3 + 0.25 this (risk wts): {sh(cm):.2f} vs xsr3 alone {sh(cm0):.2f} "
              f"(halves {sh(cm[cm.index<'2021-10-28']):.2f}/{sh(cm[cm.index>='2021-10-28']):.2f} vs {sh(cm0[cm0.index<'2021-10-28']):.2f}/{sh(cm0[cm0.index>='2021-10-28']):.2f})")
pd.DataFrame(res).to_parquet(f"{H}/x5_res.parquet")
