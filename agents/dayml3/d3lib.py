#!/usr/bin/env python3
"""dayml3 lane: the learned intraday book (decide 09:31, fill 09:33 bar = bin 3, exit 10:01 bar = bin 31) re-built for
REALISTIC execution: per-name, per-time predicted NBBO half-spreads (liq.py hs0931 / hs1001) as a COST (entry pays
f x hs0931, exit pays f x hs1001) and the liquidity proxies as FEATURES.
Frame: d2lib.build (dlib.EXT features, y_30d / y_60d / y_120d targets) + liq.add_liq columns, cut at rlib.DEV_END.
Book series: gross (mid-to-mid), spread cost at f, old-convention cost (1-1.5 bps/side), borrow, commission ($3/order on $0.4M)."""
import os, sys, time, numpy as np, pandas as pd
R = "/home/user/llm_sharpe_optimization"
for p in [R, f"{R}/agents/dayml", f"{R}/agents/dayml2"]: sys.path.insert(0, p)
import rlib, dlib, d2lib, liq
from rlib import DEV_END
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/out"; os.makedirs(OUT, exist_ok=True)
CAP, COMM = 400000.0, 3.0
LIQ_LAG = ["ldv20", "lpx", "hl1", "hl5", "hlc", "roll30", "n30", "nc30", "dv0", "dvc"]
LIQ_T = ["t_hl0", "t_hl1", "t_n0", "t_n1", "t_dv0"]
LIQ_HS = ["hs0931", "hs1001"]
LIQ = LIQ_LAG + LIQ_T + LIQ_HS
EXTL = dlib.EXT + LIQ                       # 43 + 17 = 60 features
TIERS = [(1, 300), (301, 600), (601, 1500)]

def log(msg, fn=f"{HERE}/log.txt"):
    line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True)
    with open(fn, "a") as fh: fh.write(line + "\n")

def tier(rank):
    r = np.asarray(rank, float); return np.where(r <= 300, "1-300", np.where(r <= 600, "301-600", "601-1500"))

def build(n=1500, verbose=False):
    """d2lib frame (no onml prediction) + liq columns; hs columns of day t use only bins 0-1 of day t (t_*) and lagged proxies."""
    t0 = time.time(); tks = rlib.universe(n)
    D = d2lib.build(tks, verbose=verbose, onml=False)
    D = D.drop(columns=[c for c in ["on_pred", "on_pred_rank"] if c in D.columns])
    D = liq.add_liq(D, cols=LIQ)
    D = D[D.date <= DEV_END].sort_values(["date", "tk"]).reset_index(drop=True)
    D["tier_"] = tier(D["rank"])
    for c in LIQ_HS:                                          # a few names lack the liq frame early on: cost them at the (date, tier) median
        D[c] = D[c].fillna(D.groupby(["date", "tier_"])[c].transform("median")).fillna(D[c].median()).astype(np.float32)
    D["hs_rt"] = D["hs0931"] + D["hs1001"]                     # predicted round-trip half-spread in bps (entry 09:33 ~ 09:31 model, exit 10:01)
    D["cost_old"] = 2.0 * D.tk.map(rlib.cost_bps).astype(np.float32)
    log(f"build: rows {len(D)} names {D.tk.nunique()} hs coverage {D.hs_rt.notna().mean():.3f} {time.time()-t0:.0f}s")
    return D

def xr(D, p):
    s = pd.Series(np.asarray(p, float), index=D.index); return (s.groupby(D.date).rank(pct=True) - 0.5).to_numpy()

def load_saved(D, y="y_60d"):
    return d2lib.load_preds(D, y)

def ens_saved(D):
    """the dayml2_ens book score: average per-day pct rank of the saved quarterly y_60d / y_120d predictions"""
    return (xr(D, load_saved(D, "y_60d")) + xr(D, load_saved(D, "y_120d"))) / 2

def positions(D, pred, k=8, y="y_30d", mask=None, long_score=None, short_score=None):
    """L/S top-k / bottom-k positions.  Either one ranking `pred` (top k long, bottom k short), or separate
    long_score / short_score (top k of each; used by the net-score selection).  mask: eligible rows."""
    cols = ["date", "tk", y, "hs0931", "hs1001", "hs_rt", "cost_old", "rank"]
    ok = D[y].notna().to_numpy()
    if mask is not None: ok &= np.asarray(mask, bool)
    if long_score is None:
        p = np.asarray(pred, float); ok &= np.isfinite(p)
        g = D.loc[ok, cols].copy(); g["p"] = p[ok]
        g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size")
        g = g[g.n >= k + 5]
        top = g[g.rk > g.n - k].copy(); bot = g[g.rk <= k].copy()
    else:
        ls, ss = np.asarray(long_score, float), np.asarray(short_score, float)
        ok &= np.isfinite(ls) & np.isfinite(ss)
        g = D.loc[ok, cols].copy(); g["pl"] = ls[ok]; g["ps"] = ss[ok]
        g["n"] = g.groupby("date")["tk"].transform("size"); g = g[g.n >= 2 * k + 5]
        g["rkl"] = g.groupby("date")["pl"].rank(method="first", ascending=False); g["rks"] = g.groupby("date")["ps"].rank(method="first", ascending=False)
        top = g[g.rkl <= k].copy(); bot = g[(g.rks <= k) & (g.rkl > k)].copy()
    top["w"] = 1.0 / k; bot["w"] = -1.0 / k
    L = pd.concat([top, bot]); L["r"] = L[y].astype(float); L["tier"] = tier(L["rank"])
    return L

def daily(L, k):
    """per-date frame: g gross, sc spread cost at f=1 (fraction), old old-convention cost, bor borrow, comm commission"""
    L = L.copy(); aw = L.w.abs()
    L["g_"] = L.w * L.r; L["sc_"] = aw * L.hs_rt / 1e4; L["old_"] = aw * L.cost_old / 1e4
    L["bor_"] = np.where(L.w < 0, aw, 0.0) * 0.01 / 252
    out = L.groupby("date").agg(g=("g_", "sum"), sc=("sc_", "sum"), old=("old_", "sum"), bor=("bor_", "sum"), npos=("w", "size"))
    out["comm"] = out.npos * 2 * COMM / CAP
    return out.loc[rlib.START:DEV_END]

def net(dd, f=1.0, comm=True, old=False):
    r = dd.g - (dd.old if old else f * dd.sc) - dd.bor
    if comm: r = r - dd.comm
    return r

def stats(r):
    r = r.fillna(0.0); eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    fo = r.loc["2018-07-01":]                      # single-candidate fold-OOS = the stitched 16 half-year windows
    h1 = r[r.index < "2021-10-28"]; h2 = r[r.index >= "2021-10-28"]
    yr = r.groupby(r.index.year).apply(rlib.sh)
    return dict(sh=rlib.sh(r), fo=rlib.sh(fo), ret=r.mean() * 252, vol=r.std() * np.sqrt(252), mdd=mdd, h1=rlib.sh(h1), h2=rlib.sh(h2),
                yrs=" ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()))

def ev(dd, label, fs=(1.0, 0.5, 0.25), p=True):
    """one line per f (commission included) + old convention + gross"""
    rows = []
    for f in fs:
        s = stats(net(dd, f)); s.update(label=label, f=f); rows.append(s)
    s = stats(net(dd, old=True)); s.update(label=label, f="old"); rows.append(s)
    s = stats(net(dd, 0.0)); s.update(label=label, f="gross-comm"); rows.append(s)
    if p:
        for s in rows:
            log(f"{label:38s} f={str(s['f']):10s} Sh {s['sh']:5.2f} fo {s['fo']:5.2f} ret {s['ret']*100:6.1f}% mdd {s['mdd']*100:6.1f}% halves {s['h1']:5.2f}/{s['h2']:5.2f} | {s['yrs']}")
    return rows

def tier_table(L, f=1.0):
    """per tier: share of positions, mean gross / round-trip spread cost / net bps per trade, and the Sharpe of the tier's own
    contribution (sum of w x r - cost over the positions in the tier, per day, scaled to the full book's leverage)"""
    L = L.copy(); L["g_"] = L.w * L.r * 1e4; L["c_"] = L.w.abs() * L.hs_rt * f; L["n_"] = L.g_ - L.c_
    rows = []
    for t in ["1-300", "301-600", "601-1500"]:
        x = L[L.tier == t]
        if len(x) == 0: rows.append(dict(tier=t, share=0.0)); continue
        k = 1.0 / L.w.abs().iloc[0]
        rows.append(dict(tier=t, share=len(x) / len(L), gross_bps=(x.g_ * k).mean(), cost_bps=(x.c_ * k).mean(), net_bps=(x.n_ * k).mean(),
                         hs_in=x.hs0931.median(), hs_out=x.hs1001.median()))
    return pd.DataFrame(rows)

def comm_bps(k): return 2 * k * 2 * COMM / CAP * 1e4

def save(name, r, note):
    rlib.save_cand("dayml3", name, r, note)
