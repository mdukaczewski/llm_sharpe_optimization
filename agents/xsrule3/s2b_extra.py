#!/usr/bin/env python3
"""Extras for the report: (i) the breadth check in the other direction (lo10_s15 on the 50 / 75 / 100 most liquid names, MOO c=0.3),
(ii) c-sensitivity, per-fold Sharpe and correlations for the shortlisted books, (iii) the filtered books at the old / x05 conventions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3lib import *
log("s2b_extra start")
D = load_frame(600); sig = cax(D)
for u in [50, 75, 100]:
    sub = D["rank"] <= u; P = positions(D[sub], sig[sub], k=10, mode="lo", smin=1.5)
    log(f"--- lo10_s15 u{u}: picks/day {len(P)/P.date.nunique():.1f} gross y_oo {np.mean(P.y_oo)*1e4:.1f} bps cost/side in {cost_sides(P,0.3)[0].mean():.2f} out {cost_sides(P,0.3)[1].mean():.2f}")
    for ex, kw in [("moo_c03", dict(exit="oo", c=0.3)), ("moo_c01", dict(exit="oo", c=0.1)), ("x05", dict(exit="on", f=0.5)), ("old1bp", dict(old=True))]:
        rep(net(P, **kw), f"lo10_s15_u{u}_{ex}", 10)
SHORT = [("lo10_s15_u150_base", 150, 10, "lo", 1.5, None), ("lo10_s15_u150_fhs10", 150, 10, "lo", 1.5, ("hs", 10)), ("lo10_s15_u150_fac5", 150, 10, "lo", 1.5, ("ac", 5)),
         ("lo10_s15_u300_fac5", 300, 10, "lo", 1.5, ("ac", 5)), ("lo10_s15_u300_fhs10", 300, 10, "lo", 1.5, ("hs", 10)), ("lo20_s15_u150_base", 150, 20, "lo", 1.5, None),
         ("ls10_u150_base", 150, 10, "ls", None, None), ("ls10_u150_fhs10", 150, 10, "ls", None, ("hs", 10)), ("ls10_u150_fhs20", 150, 10, "ls", None, ("hs", 20))]
S = {}; F = {}
for name, u, k, mode, smin, flt in SHORT:
    sub = D["rank"] <= u; npos = k * (2 if mode == "ls" else 1); filt = None
    if flt and flt[0] == "hs": filt = D.hs0931 <= flt[1]
    if flt and flt[0] == "ac":
        order = CAP / npos; rv = D.rv20 / np.sqrt(252)
        filt = pd.Series(np.maximum(liq.auction_cost_bps(order, np.maximum(np.exp(D.dvc), 0.05 * np.exp(D.ldv20)), rv, 0.3), liq.auction_cost_bps(order, np.exp(D.dv0), rv, 0.3)) <= flt[1], index=D.index)
    P = positions(D[sub], sig[sub], k=k, mode=mode, smin=smin, filt=filt)
    log(f"=== {name}: picks/day {len(P)/P.date.nunique():.1f} gross {np.mean(np.sign(P.w)*P.y_oo)*1e4:.1f} bps hs0931 mean {P.hs0931.mean():.1f}")
    log("tiers:\n" + tier_table(P, 0.3, "oo").round(2).to_string(index=False))
    for ex, kw in [("moo_c01", dict(exit="oo", c=0.1)), ("moo_c03", dict(exit="oo", c=0.3)), ("moo_c1", dict(exit="oo", c=1.0)), ("x1", dict(exit="on", f=1.0)), ("x05", dict(exit="on", f=0.5)), ("old1bp", dict(old=True))]:
        r = net(P, **kw); d = rep(r, f"{name}_{ex}", npos)
        if ex == "moo_c03":
            S[name] = r; oos, tab = fold_oos(r - comm_bps(npos) / 1e4, p=False); F[name] = tab.fold_sh.values
            log("   folds: " + " ".join(f"{x:+.1f}" for x in tab.fold_sh) + f"  corr {corrs(r)}")
log("per-fold Sharpe (moo_c03, at commission), half-years from 2018H2:\n" + pd.DataFrame(F, index=[w[:7] for w in tab.window]).T.round(1).to_string())
log("correlations among the shortlisted moo_c03 books:\n" + pd.DataFrame(S).corr().round(2).to_string())
pd.DataFrame(S).to_parquet(f"{OUT}/s2b_series.parquet")
log("s2b_extra done")
