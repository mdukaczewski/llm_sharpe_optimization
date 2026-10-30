#!/usr/bin/env python3
"""The agents/lmt 'limit at the bid into the close' entry with a feasible timing: limit-on-close orders sent at 15:49:00.
Positions = book (a) (MOC-only selection on the 15:49 prediction, k6, lam 3).  Long: LOC buy at the estimated 15:49 bid
round(c378 x (1 - s x hs49 / 1e4), 2); short: LOC sell at the estimated ask round(c378 x (1 + s x hs49 / 1e4), 2); hs49 =
predicted half-spread from the inputs to 15:49 (hs1559 booster; checked against the real 15:49:00 NBBO in qcheck.py).
Filled iff the official close is strictly through the limit; the fill is at the official close and pays the MOC impact (c);
exit MOO (c).  Unfilled = no position.  Series NET of impact, borrow AND $3 per filled order (2 per filled position), decision-dated.
functions only; used by final6.py."""
from book6 import *

def lmt_exec(H, L, k, c=0.3, s=1.0, hscol="hs1559_49"):
    h = H.loc[L.index]; w = L.w.values; lg = w > 0
    order = CAP / (2 * k); hc = h.copy(); hc["rv20"] = h.rv20_49
    moc = auction(hc, order, c, "close").values; moo = auction(hc, order, c, "open").values
    hs = h[hscol].values
    lim = np.where(lg, np.round(h.cb.values * (1 - s * hs / 1e4), 2), np.round(h.cb.values * (1 + s * hs / 1e4), 2))
    cl = h.c_off.values
    fill = np.where(lg, cl < lim, cl > lim) & np.isfinite(h.y_oo.values)
    P = pd.DataFrame({"date": h.date.values, "tk": h.tk.values, "tier": h.tier.values, "w": w, "y": h.y_oo.values, "c": moc + moo, "fill": fill,
                      "edge_lim": np.where(lg, lim / cl - 1, cl / lim - 1)}, index=L.index)
    P["pnl"] = np.where(P.fill, P.w * P.y - P.w.abs() * P.c / 1e4 - np.where(P.w < 0, P.w.abs(), 0) * 0.01 / 252 - 2 * ORDER_USD / CAP, 0.0)
    return P

def lmt_daily(P, dates):
    return P.groupby("date").pnl.sum().reindex(dates).fillna(0.0).loc[rlib.START:rlib.DEV_END]

def lmt_stats(P):
    f = P[P.fill]; u = P[~P.fill]; sg = np.sign(f.w); su = np.sign(u.w)
    return dict(fill=round(P.fill.mean() * 100, 1), gross_filled=round((sg * f.y).mean() * 1e4, 2), gross_unfilled=round((su * u.y).mean() * 1e4, 2),
                cost_filled=round(f.c.mean(), 2), net_filled=round(((sg * f.y) - f.c / 1e4).mean() * 1e4, 2))

def lmetrics(r):
    fo, tab = rlib.fold_oos(r, p=False); h1 = r[r.index < "2021-10-28"]; h2 = r[r.index >= "2021-10-28"]
    yr = r.groupby(r.index.year).apply(sh)
    return dict(sh=round(sh(r), 2), fold=round(sh(fo), 2), h1=round(sh(h1), 2), h2=round(sh(h2), 2), mdd=round(lib4.mdd(r) * 100, 1),
                ret=round(r.mean() * 252 * 100, 1), vol=round(r.std() * np.sqrt(252) * 100, 1), fneg=int((tab.fold_sh < 0).sum()),
                yrs=" ".join(f"{y % 100:02d}:{v:+.1f}" for y, v in yr.items()))
