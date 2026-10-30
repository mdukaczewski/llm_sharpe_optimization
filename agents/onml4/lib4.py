#!/usr/bin/env python3
"""onml4 lane: exit/entry routing, tier decay and a netted overnight book on the SAVED onml3 predictions (no LightGBM refit).
Compact frame out/G.parquet (built by step0.py): per (date t, tk) the targets from the official close of t to
  y_oo (official open t+1), y_on (09:31 bar, bin 1), y_0935 (bin 5), y_1001 (bin 31), y_1031 (bin 61) of t+1,
  r_moc = official close(t) / close of the 15:59 bar(t) - 1 (diagnostic for the 15:59:30 cross entry),
  liquidity inputs (lagged) and predicted half-spreads of day t (hs*, known by 15:59 of t: they use bins 0-1 / 360-389 of t
  and lagged means) and of day t+1 (n_hs*, the model's estimate of the spread actually paid at the exit; used ONLY to charge
  cost, never to select), the two onml3 predictions (p40 = pred_f40_on, p26 = pred_f26calx_q3) and the xsrule cAx composite
  (150 most liquid names, z-scored within day).  Everything <= rlib.DEV_END."""
import os, sys, time
import numpy as np, pandas as pd
HERE = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, HERE)
import rlib, liq
from rlib import sh
LANE = f"{HERE}/agents/onml4"; OUT = f"{LANE}/out"; os.makedirs(OUT, exist_ok=True); LOG = f"{LANE}/log.txt"
CAP = 400000.0; ORDER_USD = 3.0
TIERS = [(1, 300), (301, 600), (601, 1500), (1501, 2000)]
TNAMES = ["1-300", "301-600", "601-1500", "1501-2000"]
EXITS = {"oo": "y_oo", "0931": "y_on", "0935": "y_0935", "1001": "y_1001", "1031": "y_1031"}

def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as fh: fh.write(s + "\n")

def load_G():
    G = pd.read_parquet(f"{OUT}/G.parquet"); return G[G.date <= rlib.DEV_END].reset_index(drop=True)

# ---------------------------------------------------------------- costs (bps per side)
def auction(G, order, c, side):
    rvd = G.rv20.astype(float).clip(0.02, 3.0) / np.sqrt(252)
    if side == "open": adv = np.exp(G.dv0.astype(float))
    else: adv = np.maximum(np.exp(G.dvc.astype(float)), 0.05 * np.exp(G.ldv20.astype(float)))
    out = liq.auction_cost_bps(order, adv, rvd, c)
    return pd.Series(np.where(G.dv0.isna() | G.rv20.isna(), np.nan, out), index=G.index)

def route_costs(G, k, c=0.3, f=1.0, ls=True):
    """per-side cost columns for every leg, order = CAP / (2k) (L/S).  sel_* = known at 15:59 of t (selection);
    chg_* = charged (exit spreads use the t+1 prediction).  Returns dict name -> (sel, chg) Series in bps."""
    order = CAP / ((2 if ls else 1) * k); C = {}
    moc = auction(G, order, c, "close"); C["moc"] = (moc, moc)
    C["moc_half"] = (auction(G, order / 2, c, "close"),) * 2
    C["x1559"] = (f * G.hs1559, f * G.hs1559)                                   # cross at 15:59:30, same-day prediction
    moo = auction(G, order, c, "open"); C["moo"] = (moo, moo)
    C["moo_half"] = (auction(G, order / 2, c, "open"),) * 2
    for t, col in [("0931", "hs0931"), ("0935", "hs0935"), ("1001", "hs1001"), ("1031", "hs1001")]:   # 10:31: hs1001 as a conservative proxy
        C["x" + t] = (f * G[col], f * G["n_" + col].fillna(G[col]))
    return C

# ---------------------------------------------------------------- causal maps
def pbin(p, nb=50):
    return ((p.clip(-0.5, 0.4999) + 0.5) * nb // 1).astype(int)

def causal_cal(G, y="y_oo", p="p", nb=50, lag_days=3):
    """onml3 lib3.causal_cal: E[y | p-bin], map for year Y fitted on dates < Jan 1 Y - lag; first year uses itself (flagged)."""
    pb = pbin(G[p], nb); yrs = G.date.dt.year; ey = pd.Series(np.nan, index=G.index); ok = G[y].notna()
    for Y in sorted(yrs.unique()):
        prior = ok & (G.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=lag_days))
        if prior.sum() < 50000: prior = ok & (yrs == Y)
        m = G.loc[prior, y].groupby(pb[prior]).mean(); sel = yrs == Y; ey[sel] = pb[sel].map(m).values
    return ey.astype(np.float32)

def decay_ratio(G, y, p="p", lag_days=3, ext=0.05):
    """causal ratio sum(side*y_x) / sum(side*y_oo) over the extreme 5% of each day's prediction rank, by (tier, side),
    expanding by calendar year (year Y uses dates < Jan 1 Y - lag; first year in-sample).  Returns (ratio_long, ratio_short)
    Series aligned to G (ratio of the row's tier)."""
    hi = G[p] >= 0.5 - ext; lo = G[p] < -0.5 + ext
    yrs = G.date.dt.year; rL = pd.Series(np.nan, index=G.index); rS = pd.Series(np.nan, index=G.index)
    ok = G[y].notna() & G.y_oo.notna()
    for Y in sorted(yrs.unique()):
        prior = ok & (G.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=lag_days))
        if prior.sum() < 50000: prior = ok & (yrs == Y)
        sel = yrs == Y
        for msk, sgn, out in [(hi, 1, rL), (lo, -1, rS)]:
            m = prior & msk
            num = (sgn * G.loc[m, y]).groupby(G.loc[m, "tier"]).sum(); den = (sgn * G.loc[m, "y_oo"]).groupby(G.loc[m, "tier"]).sum()
            r = (num / den).clip(0, 1.5); out[sel] = G.loc[sel, "tier"].map(r).values
    return rL.fillna(1.0).astype(np.float32), rS.fillna(1.0).astype(np.float32)

# ---------------------------------------------------------------- generic book
def book(G, k, sL, sS, yL, yS, cL, cS, xoL=None, xoS=None, borrow=0.01, ret_pos=False, mode="ls"):
    """L/S equal-weight book: long the top k of sL, short the top k of sS per day (onml3 book3 logic).
    yL/yS: realised gross of the route of each row if long / short; cL/cS: charged cost (bps, both sides summed) per row;
    xoL/xoS: extra orders per position (split legs) -> extra commission subtracted inside the series.
    Returns the daily series NET of cost, borrow and extra commission, BEFORE the base commission (2 orders per position)."""
    idx = sL.notna() & sS.notna() & yL.notna() & yS.notna() & cL.notna() & cS.notna()
    g = pd.DataFrame({"date": G.date[idx], "tk": G.tk[idx], "tier": G.tier[idx], "sL": sL[idx], "sS": sS[idx], "yL": yL[idx], "yS": yS[idx],
                      "cL": cL[idx], "cS": cS[idx], "xoL": (xoL[idx] if xoL is not None else 0.0), "xoS": (xoS[idx] if xoS is not None else 0.0)})
    g["n"] = g.groupby("date").sL.transform("size"); g = g[g.n >= k + 5]
    g["rL"] = g.groupby("date").sL.rank(method="first", ascending=False); g["rS"] = g.groupby("date").sS.rank(method="first", ascending=False)
    top = g[g.rL <= k]; parts = [pd.DataFrame({"date": top.date, "tk": top.tk, "tier": top.tier, "w": 1.0 / k, "y": top.yL, "c": top.cL, "xo": top.xoL})]
    if mode == "ls":
        bot = g[g.rS <= k]; parts.append(pd.DataFrame({"date": bot.date, "tk": bot.tk, "tier": bot.tier, "w": -1.0 / k, "y": bot.yS, "c": bot.cS, "xo": bot.xoS}))
    L = pd.concat(parts)
    L["g"] = L.w * L.y; L["cost"] = L.w.abs() * L.c / 1e4 + L.w.clip(upper=0).abs() * borrow / 252 + L.xo * ORDER_USD / CAP
    L["net"] = L.g - L.cost
    out = L.groupby("date").agg(g=("g", "sum"), cost=("cost", "sum")); n = (out.g - out.cost).loc[rlib.START:rlib.DEV_END]
    return (n, L) if ret_pos else n

def commission(r, k, ls=True):
    return r - (2 * k if ls else k) * 2 * ORDER_USD / CAP

def mdd(r):
    eq = (1 + r.fillna(0)).cumprod(); return (eq / eq.cummax() - 1).min()

def metrics(r, k, ls=True):
    r = r.loc[rlib.START:rlib.DEV_END]; rn = commission(r, k, ls)
    fo, tab = rlib.fold_oos(rn, p=False)
    h1 = rn[rn.index < "2021-10-28"]; h2 = rn[rn.index >= "2021-10-28"]
    yr = rn.groupby(rn.index.year).apply(sh)
    return dict(sh_pre=sh(r), sh=sh(rn), fold=sh(fo), h1=sh(h1), h2=sh(h2), mdd=mdd(rn), ret=rn.mean() * 252, vol=rn.std() * np.sqrt(252),
                fneg=int((tab.fold_sh < 0).sum()), yrs=" ".join(f"{v:+.1f}" for y, v in yr.items()))

def mrow(name, m):
    return dict(book=name, pre=round(m["sh_pre"], 2), sh=round(m["sh"], 2), fold=round(m["fold"], 2), h1=round(m["h1"], 2), h2=round(m["h2"], 2),
                mdd=round(m["mdd"] * 100, 1), ret=round(m["ret"] * 100, 1), vol=round(m["vol"] * 100, 1), fneg=m["fneg"], yrs=m["yrs"])

def tier_table(L):
    rows = []; tot = L.net.sum()
    for i, nm in enumerate(TNAMES):
        x = L[L.tier == i]
        if len(x) == 0: continue
        sg = np.sign(x.w)
        rows.append(dict(tier=nm, n=len(x), pos_pct=round(len(x) / len(L) * 100, 1), gross=round((sg * x.y).mean() * 1e4, 2), cost=round(x.c.mean(), 2),
                         net=round(((sg * x.y) - x.c / 1e4).mean() * 1e4, 2), pnl_pct=round(x.net.sum() / tot * 100, 1)))
    return pd.DataFrame(rows)

def corr_with(r, s):
    j = pd.concat([r, s], axis=1).dropna(); return j.iloc[:, 0].corr(j.iloc[:, 1])

def load_ref(path):
    return pd.read_csv(f"{HERE}/{path}", index_col=0, parse_dates=True).iloc[:, 0].loc[:rlib.DEV_END]

def tlog(title, df):
    log(f"--- {title}\n" + df.to_string(index=False))

# ---------------------------------------------------------------- routes
EXIT_OPTS = ["moo", "x0935", "x1001", "x1031", "split"]          # split = half MOO + half 10:01
ENTRY_OPTS = ["moc", "x1559", "split"]                           # split = half MOC + half 15:59:30 cross
YEXIT = {"moo": "y_oo", "x0931": "y_on", "x0935": "y_0935", "x1001": "y_1001", "x1031": "y_1031"}

def prep(G, pcol, ratios=("y_on", "y_0935", "y_1001", "y_1031")):
    """rows with a prediction; p, ey (causal map of y_oo), causal decay ratios by (tier, side) for every continuous exit."""
    H = G[G[pcol].notna() & G.y_oo.notna()].reset_index(drop=True); H["p"] = H[pcol].astype(np.float32)
    H["ey"] = causal_cal(H, "y_oo")
    for y in ratios:
        rL, rS = decay_ratio(H, y); H["rL_" + y] = rL; H["rS_" + y] = rS
    H["y_split"] = 0.5 * (H.y_oo + H.y_1001)
    return H

def comm_bps(k, ls=True):
    """one extra $3 order on one position of a top-k book, in bps of that position's notional"""
    return ORDER_USD / (CAP / ((2 if ls else 1) * k)) * 1e4

def route_book(H, k, lam=3.0, entry="moc", exit="moo", c=0.3, f=1.0, c_sel=None, f_sel=None, borrow=0.01, ret_pos=False, exits=None, entries=None):
    """Cost-aware L/S book with a given entry and exit route.  entry / exit: one option name or 'route' (per name, per side:
    exit = argmax of E[gross of that exit] - (selection cost + extra commission); entry = cheapest).  Expected gross of exit x =
    ey x causal tier/side decay ratio of x (1 for MOO).  Selection score = expected gross of the chosen route - lam x its cost
    (+ borrow for shorts).  Charged cost uses (c, f) and the t+1 spread prediction; selection uses (c_sel, f_sel) (default = c, f)
    and only day-t information.  Returns the series net of route cost + borrow + extra split commission, before base commission."""
    cs = c if c_sel is None else c_sel; fs = f if f_sel is None else f_sel
    Cc = route_costs(H, k, c=c, f=f); Cs = Cc if (cs == c and fs == f) else route_costs(H, k, c=cs, f=fs); cb = comm_bps(k)
    n = len(H)
    def exit_opt(x):   # -> (E_L, E_S, sel cost, chg cost, y, extra orders)
        if x == "moo": return H.ey.values, -H.ey.values, Cs["moo"][0].values, Cc["moo"][1].values, H.y_oo.values, np.zeros(n)
        if x == "split":
            yv = "y_1001"; eL = H.ey.values * 0.5 * (1 + H["rL_" + yv].values); eS = -H.ey.values * 0.5 * (1 + H["rS_" + yv].values)
            return (eL, eS, 0.5 * Cs["moo_half"][0].values + 0.5 * Cs["x1001"][0].values + cb, 0.5 * Cc["moo_half"][1].values + 0.5 * Cc["x1001"][1].values,
                    H.y_split.values, np.ones(n))
        yv = YEXIT[x]
        return H.ey.values * H["rL_" + yv].values, -H.ey.values * H["rS_" + yv].values, Cs[x][0].values, Cc[x][1].values, H[yv].values, np.zeros(n)
    rm = H.r_moc.fillna(0).values
    def entry_opt(x):  # -> (sel cost, chg cost, extra orders, entry-price adjustment)
        # a 15:59:30 cross fills near the 15:59-bar close, not the official close: gross = (1+y)(1+r_moc)-1, r_moc = close/c389-1
        if x == "split": return 0.5 * Cs["moc_half"][0].values + 0.5 * Cs["x1559"][0].values + cb, 0.5 * Cc["moc_half"][1].values + 0.5 * Cc["x1559"][1].values, np.ones(n), 0.5 * rm
        return Cs[x][0].values, Cc[x][1].values, np.zeros(n), (rm if x == "x1559" else np.zeros(n))
    ens = (entries or ENTRY_OPTS) if entry == "route" else [entry]
    EN = [entry_opt(x) for x in ens]; es = np.stack([e[0] for e in EN], 1); es = np.where(np.isfinite(es), es, np.inf)
    ie = np.argmin(es, 1); rr = np.arange(n)
    e_sel = np.stack([e[0] for e in EN], 1)[rr, ie]; e_chg = np.stack([e[1] for e in EN], 1)[rr, ie]; e_xo = np.stack([e[2] for e in EN], 1)[rr, ie]; e_dy = np.stack([e[3] for e in EN], 1)[rr, ie]
    exs = (exits or EXIT_OPTS) if exit == "route" else [exit]
    EX = [exit_opt(x) for x in exs]
    out = {}
    for side, j in [("L", 0), ("S", 1)]:
        val = np.stack([o[j] - o[2] / 1e4 for o in EX], 1); val = np.where(np.isfinite(val), val, -np.inf)
        ix = np.argmax(val, 1)
        pick = lambda m: np.stack([o[m] for o in EX], 1)[rr, ix]
        E = pick(j); csel = pick(2) + e_sel; cchg = pick(3) + e_chg; yv = (1 + pick(4)) * (1 + e_dy) - 1; xo = pick(5) + e_xo
        s = E - lam * csel / 1e4 - (borrow / 252 if side == "S" else 0.0)
        out[side] = (pd.Series(s), pd.Series(yv), pd.Series(cchg), pd.Series(xo), pd.Series(ix), pd.Series(ie))
    res = book(H, k, out["L"][0], out["S"][0], out["L"][1], out["S"][1], out["L"][2], out["S"][2], out["L"][3], out["S"][3], borrow=borrow, ret_pos=ret_pos)
    if not ret_pos: return res
    r, L = res
    # attach the chosen exit / entry per position
    L = L.copy(); idx = L.index.values
    L["exit"] = np.where(L.w > 0, out["L"][4].values[idx], out["S"][4].values[idx]); L["exit"] = L["exit"].map(dict(enumerate(exs)))
    L["entry"] = pd.Series(ie[idx], index=L.index).map(dict(enumerate(ens)))
    return r, L
