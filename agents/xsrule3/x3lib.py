"""xsrule3 lane helpers: the cAx rule book re-built for realistic execution (MOC entry, official-open exit).
All loaders go through rlib (cut at rlib.DEV_END); never sets STRAT_END.  No model fitting, single core.

Frame: rlib.minute_frame(universe(1500)) + xlib extras (vlast_x, cross-sectional rank over the 1500 names, exactly as
agents/xsrule/final.py and k10.py computed the saved candidates) + liq.add_y_oo (y_oo) + liq.add_liq (dv0, dvc, ldv20,
hs0931, hs1559, hl1), then cut to universe(600) and cached under agents/xsrule3/out/ (gitignored).

Execution-cost model per (date, tk) at order size `order` dollars (= $400k / positions):
  entry (MOC, close of t):   liq.auction_cost_bps(order, max(exp(dvc), 0.05*exp(ldv20)), rv20/sqrt(252), c)
  exit  (MOO, open of t+1):  liq.auction_cost_bps(order, exp(dv0), rv20/sqrt(252), c)             [exit='oo', target y_oo]
  exit  (09:31 bar):         f x hs0931 (predicted NBBO half-spread at 09:31, f = 1 crosses, 0.5 half)  [exit='on', target y_on]
  old convention:            rlib.cost_bps(tk) per side both legs (1 bp on ranks <= 600), target y_on
All liquidity inputs are lagged 20-day means (known before the open of t); rv20 is the 20-day sd of close-to-close
returns up to the close of t (cost calibration only, not a signal)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/agents/xsrule")
import numpy as np, pandas as pd
import rlib, liq
from rlib import sh, stats, fold_oos, DEV_END, START
import xlib
LANE = "xsrule3"; OUT = f"{HERE}/out"; os.makedirs(OUT, exist_ok=True)
CAP = 400_000.0; COMM = 3.0
LIQ_COLS = ["ldv20", "dv0", "dvc", "hs0931", "hs1559", "hl1"]
KEEP = ["date", "tk", "z30", "z60", "z5l", "zon_prev", "rv20", "y_on", "y_cc", "vlast_x", "r_vlast_x"]

def log(msg):
    with open(f"{HERE}/log.txt", "a") as fh: fh.write(f"{pd.Timestamp.now():%m-%d %H:%M:%S} {msg}\n")
    print(msg, flush=True)

def load_frame(n=600, verbose=True):
    cp = f"{OUT}/D_u{n}_{DEV_END}.parquet"
    if os.path.exists(cp):
        D = pd.read_parquet(cp)
        if verbose: log(f"frame cached {cp}: rows {len(D)} names {D.tk.nunique()}")
        return D
    t0 = time.time(); tks = rlib.universe(1500)
    D = rlib.add_market(rlib.minute_frame(tks, verbose=False)); D = xlib.add_extras(D); D = xlib.add_extra2(D, verbose=False)
    D = rlib.clean(D, ["z30", "z60", "z5l", "zon_prev"])
    D["rank"] = D.tk.map(rlib.RANK).astype(int)
    D = D[D["rank"] <= n][KEEP + ["rank"]].reset_index(drop=True)
    D = liq.add_y_oo(D); D = liq.add_liq(D, cols=LIQ_COLS)
    D = D.sort_values(["date", "tk"]).reset_index(drop=True)
    D.to_parquet(cp); log(f"frame built on universe(1500) -> cut to {n}: rows {len(D)} names {D.tk.nunique()} {time.time()-t0:.0f}s")
    return D

def cax(D):
    """cAx = 0.5*z30 - z60 - 0.5*z5l + 0.3*zon_prev - rank(vlast_x), fixed weights (agents/xsrule)."""
    return 0.5 * D.z30 - D.z60 - 0.5 * D.z5l + 0.3 * D.zon_prev - 1.0 * D.r_vlast_x

def positions(D, sig, k=10, mode="lo", smin=None, filt=None):
    """Top-k (and bottom-k for 'ls') picks per date from `sig` (Series aligned to D.index) inside the rows of D
    (optionally restricted by the boolean mask `filt`).  Returns the picked rows with weight w = +-1/k."""
    ok = sig.notna() & D.y_oo.notna() & D.y_on.notna()
    if filt is not None: ok &= filt.fillna(False)
    g = D.loc[ok, ["date", "tk", "rank", "y_oo", "y_on", "rv20", "dv0", "dvc", "ldv20", "hs0931"]].copy(); g["p"] = sig[g.index].values
    g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size"); g = g[g.n >= k + 5]
    top = g[g.rk > g.n - k]; bot = g[g.rk <= k] if mode == "ls" else g.iloc[0:0]
    if smin is not None: top = top[top.p > smin]
    P = pd.concat([top.assign(w=1.0 / k), bot.assign(w=-1.0 / k)])
    P["npos"] = k * (2 if mode == "ls" else 1)
    return P.reset_index(drop=True)

def cost_sides(P, c=0.3, f=1.0, exit="oo", cap=CAP):
    """Per-side cost columns (bps of notional) for the picks: c_in (MOC entry) and c_out (exit)."""
    order = cap / P.npos.values; rv = P.rv20.values / np.sqrt(252)
    dvc = np.maximum(np.exp(P.dvc.values), 0.05 * np.exp(P.ldv20.values)); dv0 = np.exp(P.dv0.values)
    c_in = liq.auction_cost_bps(order, dvc, rv, c)
    if exit == "oo": c_out = liq.auction_cost_bps(order, dv0, rv, c)
    elif exit == "on": c_out = liq.spread_cost_bps(P.hs0931.values, f)
    else: raise ValueError(exit)
    return np.nan_to_num(c_in, nan=1.0), np.nan_to_num(c_out, nan=1.0)

def net(P, exit="oo", c=0.3, f=1.0, old=False, borrow=0.01, cap=CAP, detail=False):
    """Daily net series of the picks (BEFORE commission): gross - |w| x (c_in + c_out)/1e4 - borrow on shorts.
    old=True: the project's old convention (rlib.cost_bps per side both legs, 09:31 exit)."""
    y = P.y_on.values if (exit == "on" or old) else P.y_oo.values
    if old: c_in = c_out = P.tk.map(rlib.cost_bps).values.astype(float)
    else: c_in, c_out = cost_sides(P, c, f, exit, cap)
    w = P.w.values
    L = pd.DataFrame({"date": P.date, "g": w * y, "ci": np.abs(w) * c_in / 1e4, "co": np.abs(w) * c_out / 1e4,
                      "b": np.clip(w, None, 0) * -borrow / 252})
    out = L.groupby("date").sum(); n = (out.g - out.ci - out.co - out.b).loc[START:DEV_END]
    return (n, out) if detail else n

def comm_bps(npos, cap=CAP, comm=COMM):
    return npos * 2 * comm / cap * 1e4

def rep(r, name, npos, p=True):
    """Stats at the $3/$0.4M commission: full Sh, fold-OOS, halves, maxDD, plus pre-commission Sh."""
    rc = r - comm_bps(npos) / 1e4
    s = stats(rc, name, p=False); oos, tab = fold_oos(rc, p=False)
    d = dict(name=name, sh_pre=sh(r), sh=s["sharpe"], oos=sh(oos), h1=s["h1"], h2=s["h2"], mdd=s["mdd"], ret=s["ret"], vol=s["vol"],
             fmin=tab.fold_sh.min(), fneg=int((tab.fold_sh < 0).sum()), days=len(r))
    if p: log(f"{name:42s} Sh {d['sh']:5.2f} (pre-comm {d['sh_pre']:5.2f}) OOS {d['oos']:5.2f} halves {d['h1']:5.2f}/{d['h2']:5.2f} mdd {d['mdd']*100:6.1f}% ret {d['ret']*100:5.1f}% vol {d['vol']*100:4.1f}% worst fold {d['fmin']:+.1f} neg {d['fneg']}/{len(tab)}")
    return d

_refs = None
def refs():
    global _refs
    if _refs is None:
        _refs = {}
        for nm, f in [("xsr_lo10_old", "agents/xsrule/cands/xsr_cAx_lo10_s15_u150.csv"), ("onml_k10", "agents/onml/cands/onml_f26calx_u2000_q3_k10.csv")]:
            _refs[nm] = pd.read_csv(f"{ROOT}/{f}", index_col=0, parse_dates=True).iloc[:, 0].loc[:DEV_END]
        _refs["spy_cc"] = rlib.ticker_frame("SPY").loc[START:DEV_END].y_cc.astype(float)
    return _refs

def corrs(r):
    R = refs(); return {k: round(float(r.corr(v.reindex(r.index))), 2) for k, v in R.items()}

def tier_table(P, c=0.3, exit="oo", tiers=((1, 50), (51, 100), (101, 150), (151, 300), (301, 600))):
    """Mean gross bps per trade, entry/exit cost per side and share of picks by liquidity tier (universe rank)."""
    y = (P.y_on if exit == "on" else P.y_oo).values; c_in, c_out = cost_sides(P, c, 1.0 if exit == "on" else 1.0, exit)
    sg = np.sign(P.w.values); rows = []
    for lo, hi in tiers:
        m = (P["rank"].values >= lo) & (P["rank"].values <= hi)
        if m.sum() == 0: continue
        rows.append(dict(tier=f"{lo}-{hi}", picks=int(m.sum()), share=m.mean(), gross_bps=np.nanmean(sg[m] * y[m]) * 1e4, c_in=c_in[m].mean(), c_out=c_out[m].mean(), hs0931=np.nanmean(P.hs0931.values[m])))
    return pd.DataFrame(rows)
