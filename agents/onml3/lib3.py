#!/usr/bin/env python3
"""onml3 lane: the learned overnight book under REALISTIC execution (MOC entry + MOO exit, auction impact + borrow;
or MOC entry + continuous exit paying a fraction of the predicted 09:31 half-spread).
Frame = agents/onml/olib.load(2000) + ex-auction volume features (final2000.add_x) + liq.add_liq + liq.add_y_oo,
cached in agents/onml3/out/frame.parquet.  Every number is on dates <= rlib.DEV_END."""
import os, sys, time
import numpy as np, pandas as pd
HERE = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/agents/onml")
import rlib, liq
from rlib import FEATS18, sh, Min
import olib
from olib import EXTRA10, CAL, xrank
LANE = f"{HERE}/agents/onml3"; OUT = f"{LANE}/out"; os.makedirs(OUT, exist_ok=True)
LOG = f"{LANE}/log.txt"; olib.LOG = LOG          # olib.load logs into this lane, not agents/onml
CAP = 400000.0; ORDER_USD = 3.0
PRED_OLD = f"{HERE}/agents/onml/out/pred_f26calx_u2000_q3.parquet"
CAND_OLD = f"{HERE}/agents/onml/cands/onml_f26calx_u2000_q3_k10.csv"
LIQF = ["ldv20", "lpx", "hl1", "hl5", "hlc", "roll30", "n30", "nc30", "dv0", "dvc", "c_hlc", "c_nc30", "hs0931", "hs1559"]
FS = FEATS18 + EXTRA10 + CAL
F26 = [f for f in FS if f not in ("zmoc", "vmoc_share", "z5l")]
XMAP = {"vlast_share": "vlast_x", "vday_rel": "vday_rel_x", "ldv": "ldv_x"}
F26X = [XMAP.get(f, f) for f in F26]
TIERS = [(1, 300), (301, 600), (601, 1500), (1501, 2000)]

def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as fh: fh.write(s + "\n")

# ---------------------------------------------------------------- frame
XD = f"{rlib.DATA}/cache_x"
def exauction(tk):          # copy of agents/onml/final2000.exauction (that file runs on import)
    cp = f"{XD}/{tk}.parquet"; src = f"{rlib.DATA}/minute_npz/{tk}.npz"
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(src): return pd.read_parquet(cp)
    M = Min(tk, end="2026-10-28"); v = M.v; vday = v[:, :390].sum(1)
    d = pd.DataFrame({"vlast_x": v[:, 330:390].sum(1) / (vday + 1), "vday_rel_x": vday / (pd.Series(vday).rolling(20).mean().shift(1).values + 1),
                      "ldv_x": np.log(vday * M.close + 1)}, index=M.dates).astype(np.float32); d.index.name = "date"
    d.to_parquet(cp); return d

def add_x(D):
    parts = []
    for tk in D.tk.unique():
        e = exauction(tk).reset_index(); e["tk"] = tk; parts.append(e)
    return D.merge(pd.concat(parts), on=["date", "tk"], how="left")

def tier_of(tk):
    r = rlib.RANK.get(tk, 9999)
    for i, (a, b) in enumerate(TIERS):
        if a <= r <= b: return i
    return 3

def frame(n=2000, rebuild=False):
    cp = f"{OUT}/frame_u{n}.parquet"
    if os.path.exists(cp) and not rebuild:
        D = pd.read_parquet(cp); log(f"frame: cached {len(D)} rows {D.tk.nunique()} names {D.date.min().date()}..{D.date.max().date()}"); return D
    t0 = time.time(); D = olib.load(n); D = add_x(D); D = rlib.clean(D, ["vlast_x", "vday_rel_x", "ldv_x"])
    D = liq.add_liq(D); D = liq.add_y_oo(D)
    D = D[D.date <= rlib.DEV_END].reset_index(drop=True)
    D["rank"] = D.tk.map(rlib.RANK).fillna(9999).astype(np.int32); D["tier"] = D.tk.map(tier_of).astype(np.int8)
    D = fix_liq(D)
    # targets: per-day ranks of y_oo and of y_oo net of the predicted exit (MOO) auction cost at c=0.3 (see cost_cols)
    cost_cols(D, k=10, c=0.3)
    D["yr_oo"] = (D.groupby("date").y_oo.rank(pct=True) - 0.5).astype(np.float32)
    D["y_oon"] = D.y_oo - D.cx / 1e4
    D["yr_oon"] = (D.groupby("date").y_oon.rank(pct=True) - 0.5).astype(np.float32)
    D.to_parquet(cp); log(f"frame built {len(D)} rows {D.tk.nunique()} names {time.time()-t0:.0f}s mem {D.memory_usage().sum()/1e9:.2f}GB")
    return D

def fix_liq(D):
    """liq columns are NOT passed through rlib.clean (it clips to +-5; ldv20/dv0/dvc are ~15-23, hs0931 is in bps):
    only +-inf -> NaN (LightGBM handles NaN natively)."""
    D[LIQF] = D[LIQF].replace([np.inf, -np.inf], np.nan).astype(np.float32); return D

def remerge_liq(n=2000):
    """repair a cached frame whose LIQF columns were clipped: drop and re-merge them from data/cache_liq"""
    cp = f"{OUT}/frame_u{n}.parquet"; D = pd.read_parquet(cp); t0 = time.time()
    D = D.drop(columns=[c for c in LIQF + ["ce", "cx", "y_oon", "yr_oon"] if c in D.columns])
    D = liq.add_liq(D, cols=LIQF); D = fix_liq(D); cost_cols(D, k=10, c=0.3)
    D["y_oon"] = D.y_oo - D.cx / 1e4; D["yr_oon"] = (D.groupby("date").y_oon.rank(pct=True) - 0.5).astype(np.float32)
    D.to_parquet(cp); log(f"remerge_liq done {time.time()-t0:.0f}s; ldv20 median {D.ldv20.median():.2f} hs0931 median {D.hs0931.median():.2f} cx median {D.cx.median():.2f}")
    return D

# ---------------------------------------------------------------- costs
def cost_cols(D, k=10, c=0.3):
    """adds ce (MOC entry) / cx (MOO exit) auction cost in bps per side for an L/S top-k book on CAP equity.
    MOO: expected auction $vol = exp(dv0) (lagged 20d mean of the 09:30 bar); MOC: max(exp(dvc), 5% of 20d mean daily $vol)
    (the 16:00 bar understates the closing auction).  rv20 of the rlib frame is annualised -> daily."""
    order = CAP / (2 * k); rvd = D.rv20.astype(float).clip(0.02, 3.0) / np.sqrt(252)
    adv_open = np.exp(D.dv0.astype(float)); adv_close = np.maximum(np.exp(D.dvc.astype(float)), 0.05 * np.exp(D.ldv20.astype(float)))
    D["ce"] = liq.auction_cost_bps(order, adv_close, rvd, c).astype(np.float32)
    D["cx"] = liq.auction_cost_bps(order, adv_open, rvd, c).astype(np.float32)
    D.loc[D.dv0.isna() | D.rv20.isna(), ["ce", "cx"]] = np.nan
    return D

def commission(r, k, ls=True):
    """subtract the $3/order commission: 2k positions x 2 orders x $3 per day on CAP equity (3 bps/day for k=10)"""
    return r - (2 * k if ls else k) * 2 * ORDER_USD / CAP

# ---------------------------------------------------------------- books
def book3(G, k=10, y="y_oo", ce="ce", cx="cx", score="p", borrow=0.01, cap=None, capcol=None, mode="ls", ret_pos=False, score_s=None):
    """Daily return of an L/S top-k equal-weight book from a frame G with columns date, tk, score, y, ce, cx (bps per side).
    cap/capcol: keep only names with capcol <= cap before ranking (cost-aware universe).  Returns the series NET of
    ce + cx + borrow but BEFORE commission (use commission(r, k))."""
    cols = list(dict.fromkeys(["date", "tk", y, "tier"] + [c for c in (ce, cx, capcol) if c is not None]))
    g = G.loc[G[score].notna() & G[y].notna(), cols].copy(); g["p"] = G.loc[g.index, score].values
    if score_s is not None: g["ps"] = G.loc[g.index, score_s].values      # separate score for the short side (higher = better short)
    if cap is not None: g = g[g[capcol] <= cap]
    g["c"] = (g[ce] if ce is not None else 0.0) + (g[cx] if cx is not None else 0.0)
    g = g[g.c.notna()]
    g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size")
    g = g[g.n >= k + 5]
    top = g[g.rk > g.n - k].copy(); top["w"] = 1.0 / k
    if mode == "ls" and score_s is None: bot = g[g.rk <= k].copy(); bot["w"] = -1.0 / k
    elif mode == "ls":
        g["rks"] = g.groupby("date")["ps"].rank(method="first"); bot = g[g.rks > g.n - k].copy(); bot["w"] = -1.0 / k
    else: bot = g.iloc[0:0].copy()
    L = pd.concat([top, bot]); L["g"] = L.w * L[y]
    L["cost"] = L.w.abs() * L.c / 1e4 + L.w.clip(upper=0).abs() * borrow / 252
    L["net"] = L.g - L.cost
    out = L.groupby("date").agg(g=("g", "sum"), cost=("cost", "sum")); n = (out.g - out.cost).loc[rlib.START:rlib.DEV_END]
    return (n, L) if ret_pos else n

def mdd(r):
    eq = (1 + r.fillna(0)).cumprod(); return (eq / eq.cummax() - 1).min()

def metrics(r, k, ls=True):
    """at the $0.4M commission: full Sharpe, fold-OOS, halves, maxDD, annual ret; also spread-only (pre-commission) Sharpe"""
    r = r.loc[rlib.START:rlib.DEV_END]; rn = commission(r, k, ls)
    fo = rlib.fold_oos(rn, p=False)[0]
    h1 = rn[rn.index < "2021-10-28"]; h2 = rn[rn.index >= "2021-10-28"]
    yr = rn.groupby(rn.index.year).apply(sh)
    return dict(sh_pre=sh(r), sh=sh(rn), fold=sh(fo), h1=sh(h1), h2=sh(h2), mdd=mdd(rn), ret=rn.mean() * 252, vol=rn.std() * np.sqrt(252),
                yrs=" ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()))

def mline(name, m):
    return (f"{name:38s} pre-comm Sh {m['sh_pre']:5.2f} | $0.4M Sh {m['sh']:5.2f} foldOOS {m['fold']:5.2f} halves {m['h1']:.2f}/{m['h2']:.2f} "
            f"mdd {m['mdd']*100:6.1f}% ret {m['ret']*100:5.1f}% vol {m['vol']*100:4.1f}% | {m['yrs']}")

def tier_table(L, y="y_oo"):
    """positions by liquidity tier: share of positions, gross / cost / net bps per trade, share of total net PnL"""
    rows = []
    tot = L.net.sum()
    for i, (a, b) in enumerate(TIERS):
        x = L[L.tier == i]
        if len(x) == 0: rows.append((f"{a}-{b}", 0, 0, 0, 0, 0, 0)); continue
        sgn = np.sign(x.w)
        rows.append((f"{a}-{b}", len(x), len(x) / len(L) * 100, (sgn * x[y]).mean() * 1e4, x.c.mean(), ((sgn * x[y]) - x.c / 1e4).mean() * 1e4, x.net.sum() / tot * 100))
    return pd.DataFrame(rows, columns=["tier", "n", "pos%", "gross_bps", "cost_bps", "net_bps", "pnl%"])

def load_pred(path=PRED_OLD):
    P = pd.read_parquet(path); return P[P.date <= rlib.DEV_END]

def old_cand():
    return pd.read_csv(CAND_OLD, index_col=0, parse_dates=True).iloc[:, 0].loc[:rlib.DEV_END]

def corr_with(r, s):
    j = pd.concat([r, s], axis=1).dropna(); return j.iloc[:, 0].corr(j.iloc[:, 1])

def save_pred(D, p, name):
    pd.DataFrame({"date": D.date, "tk": D.tk, "p": p}).dropna().to_parquet(f"{OUT}/pred_{name}.parquet")

# ---------------------------------------------------------------- cost-aware selection (step 2)
def causal_cal(G, y="y_oo", nb=50, lag_days=3, first_year_insample=True):
    """E[y | p-bin] with an EXPANDING map: the map used in calendar year Y is fitted on dates < Jan 1 of Y minus lag_days.
    For the first year (no prior predictions) the map of that year itself is used when first_year_insample (flagged in the report;
    fold-OOS starts 2018-07 so it never touches a fold).  Returns the expected gross overnight return (fraction) per row."""
    pb = ((G.p.clip(-0.5, 0.4999) + 0.5) * nb // 1).astype(int); yrs = G.date.dt.year; ey = pd.Series(np.nan, index=G.index)
    ok = G[y].notna()
    for Y in sorted(yrs.unique()):
        prior = ok & (G.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=lag_days))
        if prior.sum() < 50000:
            if not first_year_insample: continue
            prior = ok & (yrs == Y)
        m = G.loc[prior, y].groupby(pb[prior]).mean(); sel = yrs == Y; ey[sel] = pb[sel].map(m).values
    return ey.astype(np.float32)

def net_scores(G, ey, lam, ce="ce", cx="cx", borrow=0.01):
    """long score = E[gross] - lam x (entry + exit cost); short score = -E[gross] - lam x cost - borrow/day"""
    c = (G[ce] + G[cx]) / 1e4
    G["sL"] = ey - lam * c; G["sS"] = -ey - lam * c - borrow / 252; return G
