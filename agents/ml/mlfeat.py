"""ML agent: lean loader + hand-made daily features for cross-sectional models.
Everything here is computed from data <= t (no forward bars except the label)."""
import numpy as np, pandas as pd, os, time, gc
from bt import *
from feats import *
from ls import extra_feats, backtest_W, sh

PANEL_LIQ = "/home/trumpet/strat20/data/panel_liquid.npz"

def load_sub(min_adv=50e6, min_px=5.0, path=PANEL_LIQ):
    """Load the liquid panel and keep only tickers that ever satisfy the
    universe screen (plus SPY). Returns Panel-like object with subset arrays."""
    t0 = time.time()
    P = Panel(path=path)
    is_cs = np.isin(P.ttype, ["CS", "ADRC"])
    adv20 = rolling_mean(P.dollar_vol, 20, 15)
    ok = is_cs[None, :] & (adv20 >= min_adv) & (P.close >= min_px)
    keep = ok.any(axis=0)
    keep[P.col("SPY")] = True
    del adv20, ok
    for k in ["close_raw", "close", "open", "high", "low", "vw", "vol", "n", "div", "ret", "dollar_vol"]:
        setattr(P, k, np.ascontiguousarray(getattr(P, k)[:, keep]))
    P.tickers = P.tickers[keep]; P.ttype = P.ttype[keep]
    P.T, P.N = P.close.shape
    P.ix = {t: i for i, t in enumerate(P.tickers)}
    gc.collect()
    print(f"load_sub: kept {P.N} tickers, {P.T} days, {time.time()-t0:.0f}s", flush=True)
    return P

def xs_rank_rows(a, m):
    """cross-sectional rank in [-0.5,0.5] inside mask m; 0 elsewhere. vectorised via pandas."""
    df = pd.DataFrame(np.where(m & np.isfinite(a), a, np.nan))
    r = df.rank(axis=1, pct=True).values
    cnt = df.notna().sum(axis=1).values[:, None]
    out = r - 0.5 - 0.5 / np.maximum(cnt, 1)
    return np.nan_to_num(out).astype(np.float32)

def build_features(P, F, m):
    """returns dict name -> T x N float32 rank-normalised features."""
    T, N = P.T, P.N
    logc = F["logc"]; dret = F["dret"]; spy = F["spy_ret"]
    def lag(k, skip=0):
        out = np.full((T, N), np.nan); out[k:] = logc[k - skip:T - skip] - logc[:T - k]; return out
    def rmean(a, w, mp=None): return pd.DataFrame(a).rolling(w, min_periods=mp or w).mean().values
    def rsum(a, w, mp=None): return pd.DataFrame(a).rolling(w, min_periods=mp or w).sum().values
    def rstd(a, w, mp=None): return pd.DataFrame(a).rolling(w, min_periods=mp or w).std().values
    raw = {}
    for k in [1, 2, 3, 5, 10, 21, 42, 63, 126, 252]:
        raw[f"r{k}"] = lag(k)
    raw["mom12_1"] = lag(252, 21); raw["mom6_1"] = lag(126, 21)
    # beta / residual
    W = 252
    mx = pd.Series(spy).rolling(W, min_periods=200).mean().values[:, None]
    my = rmean(dret, W, 200); mxy = rmean(dret * spy[:, None], W, 200)
    vx = pd.Series(spy).rolling(W, min_periods=200).var().values[:, None]
    beta = np.clip(np.nan_to_num((mxy - mx * my) / (vx + 1e-12)), -1, 4)
    resid = dret - beta * spy[:, None]
    raw["beta"] = beta
    for k in [1, 3, 5, 10, 21, 63]:
        raw[f"res{k}"] = rsum(resid, k)
    raw["resmom"] = rsum(resid, 252, 200) - rsum(resid, 21)
    raw["ivol60"] = rstd(resid, 60, 40)
    raw["vol5"] = rstd(dret, 5); raw["vol20"] = F["vol20"]; raw["vol60"] = F["vol60"]
    raw["volr5_60"] = raw["vol5"] / (F["vol60"] + 1e-9)
    raw["volr20_120"] = F["vol20"] / (F["vol120"] + 1e-9)
    dv = P.dollar_vol
    raw["dv1_adv20"] = dv / (F["adv20"] + 1)
    raw["adv5_60"] = rmean(dv, 5, 3) / (F["adv60"] + 1)
    raw["adv20_60"] = F["adv20"] / (F["adv60"] + 1)
    raw["ladv"] = np.log(F["adv20"] + 1)
    raw["n5_60"] = rmean(P.n, 5, 3) / (rmean(P.n, 60, 40) + 1)
    raw["n1_20"] = P.n / (rmean(P.n, 20, 15) + 1)
    co, oc = F["co"], F["oc"]
    for k in [1, 5, 21]:
        raw[f"co{k}"] = rsum(co, k) if k > 1 else co
        raw[f"oc{k}"] = rsum(oc, k) if k > 1 else oc
    hl = (P.high - P.low) / (P.close + 1e-9)
    rp = (P.close - P.low) / (P.high - P.low + 1e-9)
    raw["rp1"] = rp; raw["rp5"] = rmean(rp, 5, 3)
    raw["hl1"] = hl; raw["hl5"] = rmean(hl, 5, 3); raw["hl21"] = rmean(hl, 21, 15)
    vwd = P.close / P.vw - 1
    raw["vwd1"] = vwd; raw["vwd5"] = rmean(vwd, 5, 3)
    raw["hi52"] = F["closef"] / F["hi252"]
    raw["lo52"] = F["closef"] / pd.DataFrame(F["closef"]).rolling(252, min_periods=200).min().values
    raw["ma20"] = F["closef"] / rmean(F["closef"], 20, 15) - 1
    raw["ma50"] = F["closef"] / F["ma50"] - 1
    raw["ma200"] = F["closef"] / F["ma200"] - 1
    raw["amihud"] = rmean(np.abs(dret) / (dv + 1), 21, 15)
    raw["max21"] = pd.DataFrame(dret).rolling(21).max().values
    raw["min21"] = pd.DataFrame(dret).rolling(21).min().values
    raw["skew21"] = pd.DataFrame(dret).rolling(21).skew().values
    raw["updays10"] = rsum((dret > 0).astype(float), 10)
    raw["r1_x_dv"] = dret * raw["dv1_adv20"]          # volume-weighted reversal
    raw["r5_x_dv"] = rsum(dret * raw["dv1_adv20"], 5)
    raw["abs_r1"] = np.abs(dret)
    # dividends upcoming? no. price level
    raw["lpx"] = np.log(F["closef"])
    feats = {}
    for k, v in raw.items():
        feats[k] = xs_rank_rows(v, m)
    del raw; gc.collect()
    # market-level (same for all names), not ranked
    spy_c = F["spy_c"]
    mk = {}
    mk["m_r5"] = np.log(spy_c / np.roll(spy_c, 5)); mk["m_r21"] = np.log(spy_c / np.roll(spy_c, 21))
    mk["m_vol20"] = pd.Series(spy).rolling(20).std().values
    mk["m_ma200"] = spy_c / F["spy_ma200"] - 1
    disp = np.nanstd(np.where(m, dret, np.nan), axis=1)
    mk["m_disp5"] = pd.Series(disp).rolling(5).mean().values
    for k, v in mk.items():
        feats[k] = np.repeat(np.nan_to_num(v).astype(np.float32)[:, None], N, axis=1)
    return feats

def fwd_target(P, F, m, h, kind="cc"):
    """h-day forward return starting at close t (cc) or at open t+1 (oc),
    rank-normalised cross-sectionally. NaN where unavailable."""
    T, N = P.T, P.N
    dret = F["dret"]
    cum = np.cumsum(np.log1p(np.clip(dret, -0.95, 10)), axis=0)
    fwd = np.full((T, N), np.nan)
    fwd[:T - h] = cum[h:] - cum[:T - h]
    if kind == "oc":
        # subtract the overnight t->t+1 open (we can't capture it)
        co = F["co"]
        fwd[:T - 1] = fwd[:T - 1] - np.log1p(co[1:])
    if kind == "res":
        # residual vs SPY using trailing 252d beta known at t
        spy = F["spy_ret"]
        cs = np.cumsum(np.log1p(spy)); fs = np.full(T, np.nan); fs[:T - h] = cs[h:] - cs[:T - h]
        W = 252
        rm = lambda a, w: pd.DataFrame(a).rolling(w, min_periods=200).mean().values
        mx = pd.Series(spy).rolling(W, min_periods=200).mean().values[:, None]
        vx = pd.Series(spy).rolling(W, min_periods=200).var().values[:, None]
        beta = np.clip(np.nan_to_num((rm(dret * spy[:, None], W) - mx * rm(dret, W)) / (vx + 1e-12)), -1, 4)
        fwd = fwd - beta * fs[:, None]
    # only where there is a real bar at t and t+h
    has = ~np.isnan(P.close)
    okk = has.copy(); okk[:T - h] &= has[h:]; okk[T - h:] = False
    df = pd.DataFrame(np.where(m & okk & np.isfinite(fwd), fwd, np.nan))
    r = df.rank(axis=1, pct=True).values
    cnt = df.notna().sum(axis=1).values[:, None]
    y = (r - 0.5 - 0.5 / np.maximum(cnt, 1)).astype(np.float32)
    return y, fwd.astype(np.float32)

def build_mask(F, P, min_adv, min_px=5.0):
    """shared base_mask + the price filter on the UNADJUSTED close (base_mask uses
    split-adjusted closes, which lets later reverse-split junk into the universe)."""
    m = np.zeros((P.T, P.N), bool)
    rawc = ffill(P.close_raw, 5)
    for i in range(260, P.T):
        m[i] = base_mask(F, i, min_adv, min_px) & (rawc[i] >= min_px)
    return m
