"""Exp 1: time-series overnight per ETF with s = 0.5*z30 - z60, fixed thresholds, equal risk (1/rv20 lagged)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, elib2 as E
from rlib import sh
D = E.frame(E.ALL + ["TQQQ"]); D = D[~D.half.astype(bool)]
D["s"] = 0.5 * D.z30 - D.z60
D["rvl"] = D.groupby("tk").rv20.shift(1)
P = lambda c: D.pivot(index="date", columns="tk", values=c)
S, RV = P("s"), P("rvl"); Y = {"oo": P("y_oo"), "on": P("y_on")}
E.log("=== x1: per-ETF conditional overnight (gross, bps/night; y_oo = MOC->MOO)")
rows = []
for tk in E.ALL + ["TQQQ"]:
    s = S[tk]; y = Y["oo"][tk]; y1 = Y["on"][tk]; ok = y.notna() & s.notna()
    r = dict(tk=tk, all=y[ok].mean() * 1e4, sh_all=sh(y[ok]))
    for th in [0.5, 1.0, 1.5]:
        m = ok & (s > th); r[f"L{th}"] = y[m].mean() * 1e4; r[f"nL{th}"] = int(m.sum()); r[f"t{th}"] = y[m].mean() / y[m].std() * np.sqrt(m.sum())
        m2 = ok & (s < -th); r[f"S{th}"] = y[m2].mean() * 1e4
    m = ok & (s > 1.0); r["L1_on"] = y1[m].mean() * 1e4
    h = y.index < "2021-10-28"; r["L1_h1"] = y[m & h].mean() * 1e4; r["L1_h2"] = y[m & ~h].mean() * 1e4
    rows.append(r)
T = pd.DataFrame(rows).set_index("tk").round(2); E.log(T.to_string())
E.log("pooled EQ:  " + str(T.loc[E.EQ].mean().round(2).to_dict()))
E.log("pooled OTH: " + str(T.loc[E.OTH].mean().round(2).to_dict()))

def weights(univ, th, mode, cap=0.5, gcap=2.0, unit=0.05):
    s = S[univ]; base = (unit / RV[univ]).clip(upper=cap)
    W = base.where(s > th, 0.0)
    if mode == "ls": W = W - base.where(s < -th, 0.0)
    if mode == "all": W = base.copy()
    g = W.abs().sum(1); W = W.mul(np.minimum(1.0, gcap / g.replace(0, np.nan)).fillna(1.0), axis=0)
    return W.fillna(0.0)
res = {}
E.log("=== x1 books (weight 0.05/rv20_lag per position, cap 0.5, gross cap 2; net of cost/side, 1% borrow, $3/order at $0.4M inside)")
for univ_n, univ in [("eq", E.EQ), ("all", E.ALL)]:
    for tgt in ["oo", "on"]:
        for cost in [1.5, 3.0]:
            if tgt == "on" and cost == 3.0: continue
            n, g, npos, gr = E.run_book(weights(univ, 0, "all"), Y[tgt][univ], cost); E.report(f"{univ_n}_{tgt}_c{cost}_ALLNIGHTS", n, npos, gr)
            for th in [0.5, 1.0, 1.5]:
                for mode in ["lo", "ls"]:
                    W = weights(univ, th, mode); n, g, npos, gr = E.run_book(W, Y[tgt][univ], cost)
                    nm = f"{univ_n}_{tgt}_c{cost}_s{th}_{mode}"; res[nm] = n
                    E.report(nm, n, npos, gr)
            n0, *_ = E.run_book(weights(univ, 1.0, "lo"), Y[tgt][univ], 0, 0, comm=False); E.log(f"   gross s1 lo {univ_n} {tgt}: Sh {sh(n0):.2f} mean {n0.mean()*1e4:.2f} bp/day")
pd.DataFrame(res).to_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "x1_res.parquet"))
