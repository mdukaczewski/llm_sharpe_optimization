"""Exp 4: robustness of the x1 time-series book: (a) auction-impact cost model (liq.auction_cost_bps, c=0.1/0.3/1.0) per ETF,
(b) continuous sizing clip(s,0,1.5), (c) signal frozen at 15:50 (MOC cutoff on Nasdaq), (d) overlap with xsr3 (residual, combo)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, elib2 as E, rlib, liq
from rlib import sh, zsd, Min
H = os.path.dirname(os.path.abspath(__file__))
U = E.EQ
D = E.frame(U); D = D[~D.half.astype(bool)].copy()
D["s"] = 0.5 * D.z30 - D.z60; D["rvl"] = D.groupby("tk").rv20.shift(1)
# (c) 15:50 signal: last-hour window 14:50..15:50 (bins 319..379), same z normalisation
parts = []
for tk in U:
    M = Min(tk, end="2026-10-28"); z60b = zsd(M.r(319, 379)); z30 = zsd(M.r(-1, 29))
    parts.append(pd.DataFrame({"date": pd.DatetimeIndex(M.dates), "tk": tk, "s1550": 0.5 * z30 - z60b}))
D = D.merge(pd.concat(parts), on=["date", "tk"], how="left")
D = liq.add_liq(D, ["dv0", "dvc"])
Pv = lambda c: D.pivot(index="date", columns="tk", values=c)
S, S2, RV, Y, DV0, DVC = Pv("s"), Pv("s1550"), Pv("rvl"), Pv("y_oo"), np.exp(Pv("dv0")), np.exp(Pv("dvc"))
base = (0.05 / RV).clip(upper=0.5)
def cap(W, g=2.0):
    s_ = W.abs().sum(1); return W.mul(np.minimum(1.0, g / s_.replace(0, np.nan)).fillna(1.0), axis=0).fillna(0)
W1 = cap(base.where(S > 1.0, 0.0))
E.log("=== x4 (a) auction-impact model per side for the s>1 long book: order = w x $400k; bps = max(1, c*rv_d*sqrt(order/auction $vol))")
order = W1 * 4e5; rvd = RV / np.sqrt(252)
res = {}
for c in [0.1, 0.3, 1.0]:
    cin = liq.auction_cost_bps(order.values, DVC.values, rvd.values, c=c); cout = liq.auction_cost_bps(order.values, DV0.values, rvd.values, c=c)
    CB = pd.DataFrame((cin + cout) / 2, index=W1.index, columns=W1.columns)
    held = W1 > 0
    E.log(f"   c={c}: median per-side bps on held positions {np.nanmedian(CB.values[held.values]):.2f}, mean {np.nanmean(CB.values[held.values]):.2f}; by ETF mean: "
          + " ".join(f"{t}:{np.nanmean(CB[t][held[t]]):.1f}" for t in U))
    # book with per-name per-day cost
    g = (W1 * Y.fillna(0)).sum(1); cst = (W1.abs() * 2 * CB.fillna(1.0)).sum(1) / 1e4; npos = (W1 != 0).sum(1)
    n = (g - cst - 2 * npos * E.COMM_BPS / 1e4).loc[E.START:E.DEV_END]
    E.report(f"eq_s1_lo_auction_c{c}", n, npos.loc[n.index], W1.abs().sum(1).loc[n.index]); res[f"auc_c{c}"] = n
E.log("=== x4 (b,c) variants at 1.5 bps/side")
V = {"s1_lo": W1, "s1_lo_1550sig": cap(base.where(S2 > 1.0, 0.0)), "s05_lo_1550sig": cap(base.where(S2 > 0.5, 0.0)),
     "cont_lo": cap(base * S.clip(lower=0, upper=1.5) / 1.0), "s1_lo_noFri": cap(base.where(S > 1.0, 0.0).mul(pd.Series(S.index.weekday != 4, S.index).astype(float), axis=0))}
for nm, W in V.items():
    for cost in [1.5, 3.0]:
        n, g, npos, gr = E.run_book(W, Y, cost); E.report(f"eq_{nm}_c{cost}", n, npos, gr); res[f"{nm}_c{cost}"] = n
# (d) overlap with xsr3 (decision-dated, as ours)
x = E.refs()["xsr3_lo10"]; n = res["s1_lo_c1.5"]
a = pd.concat([n, x], axis=1, keys=["e", "x"]).dropna(); b = np.polyfit(a.x, a.e, 1)[0]; resid = a.e - b * a.x
E.log(f"=== x4 (d) vs xsr3_cAx_u50_moo_lo10: corr {a.e.corr(a.x):.2f}, beta {b:.2f}, residual Sh {sh(resid):.2f} (residual mean {resid.mean()*252*100:.1f}%/yr)")
for wv in [0.0, 0.25, 0.5]:
    xv = a.x / a.x.std(); ev = a.e / a.e.std(); cmb = (1 - wv) * xv + wv * ev
    E.log(f"   risk-weight {wv:.2f} etfon + {1-wv:.2f} xsr3: Sh {sh(cmb):.2f} | halves {sh(cmb[cmb.index<'2021-10-28']):.2f}/{sh(cmb[cmb.index>='2021-10-28']):.2f}")
pd.DataFrame(res).to_parquet(f"{H}/x4_res.parquet")
