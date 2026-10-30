"""dml2: daily-bar cross-sectional ML sleeve, rebuilt for MOC execution with $3/order commission on $0.4M.

Copies (adapted) the feature / target code of agents/ml/mlfeat.py and ml_run.py; the panel is cut at
rlib.DEV_END right after loading, so nothing after 2026-04-27 is ever in memory.

Conventions
  S[t]   score computed from data up to the close of t.
  timing 'A' (default): the book traded at the official close of day t uses S[t-1] (decided after the
         close of t-1, order sent any time before the 15:50 MOC cutoff of day t).
  timing 'B0' (reference only, NOT tradable): S[t] traded at the close of t.
  Holdings h (fraction of equity) set at close t earn the close-to-close return of t+1; that PnL is
  dated t+1 (exit-dated).  Costs of the trades at close t are dated t.
"""
import os, sys, time, gc
import numpy as np, pandas as pd
HERE = "/home/user/llm_sharpe_optimization"
sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/agents/intraday")
import rlib
from rlib import DEV_END
from bt import ffill, rolling_mean
from feats import build, base_mask
from ls import extra_feats
LANE = f"{HERE}/agents/dml2"; OUT = f"{LANE}/out"; os.makedirs(OUT, exist_ok=True)
START = "2016-10-28"
CAP = 400_000.0; ORDER_USD = 3.0

def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    with open(f"{LANE}/log.txt", "a") as fh: fh.write(time.strftime("%H:%M ") + s + "\n")

class P_: pass

def load_sub(min_adv=50e6, min_px=5.0):
    """liquid panel cut at DEV_END, only tickers that ever pass the screen (+SPY)."""
    t0 = time.time()
    z = np.load(f"{rlib.DATA}/panel_liquid.npz", allow_pickle=True)
    dates = pd.to_datetime(z["dates"]); T1 = int(np.searchsorted(dates, pd.Timestamp(DEV_END), side="right"))
    tick = z["tickers"]; ttype = z["ttype"]
    adj = z["adj"][:T1].astype(np.float64); close_raw = z["close"][:T1].astype(np.float64)
    close = close_raw * adj; vol = z["vol"][:T1].astype(np.float64) / adj
    adv20 = rolling_mean(close * vol, 20, 15)
    is_cs = np.isin(ttype, ["CS", "ADRC"])
    keep = (is_cs[None, :] & (adv20 >= min_adv) & (close >= min_px)).any(axis=0)
    keep[list(tick).index("SPY")] = True
    del adv20; gc.collect()
    P = P_(); P.dates = dates[:T1]; P.tickers = tick[keep]; P.ttype = ttype[keep]
    adj = adj[:, keep]
    P.close_raw = close_raw[:, keep]; P.close = close[:, keep]; P.vol = vol[:, keep]
    del close_raw, close, vol; gc.collect()
    for k in ["open", "high", "low", "vw", "div"]:
        setattr(P, k, z[k][:T1][:, keep].astype(np.float64) * adj)
    P.n = z["n"][:T1][:, keep].astype(np.float64)
    P.T, P.N = P.close.shape
    prev = np.vstack([np.full((1, P.N), np.nan), P.close[:-1]])
    P.ret = (P.close + P.div) / prev - 1.0
    P.dollar_vol = P.close * P.vol
    P.ix = {t: i for i, t in enumerate(P.tickers)}
    P.col = lambda t: P.ix[t]
    log(f"load_sub: {P.N} tickers, {P.T} days ({P.dates[0].date()}..{P.dates[-1].date()}), {time.time()-t0:.0f}s")
    return P

def build_mask(F, P, min_adv, min_px=5.0):
    m = np.zeros((P.T, P.N), bool); rawc = ffill(P.close_raw, 5)
    for i in range(260, P.T):
        m[i] = base_mask(F, i, min_adv, min_px) & (rawc[i] >= min_px)
    return m

def xs_rank_rows(a, m):
    df = pd.DataFrame(np.where(m & np.isfinite(a), a, np.nan))
    r = df.rank(axis=1, pct=True).to_numpy()
    cnt = df.notna().sum(axis=1).to_numpy()[:, None]
    return np.nan_to_num(r - 0.5 - 0.5 / np.maximum(cnt, 1)).astype(np.float32)

def setup(min_adv=50e6):
    P = load_sub(min_adv); F = build(P); F = extra_feats(P, F); m = build_mask(F, P, min_adv)
    log("mask avg names/day", round(float(m[300:].sum(1).mean()), 1))
    return P, F, m

def build_features(P, F, m, rows):
    """agents/ml/mlfeat.build_features (market-level features dropped = the 'n' models), ranked one at a time
    and stored only on mask cells rows=(rt, ri).  Returns names, X (nrows x K float32)."""
    T, N = P.T, P.N; rt, ri = rows
    logc = F["logc"]; dret = F["dret"]; spy = F["spy_ret"]
    names, cols = [], []
    def put(k, v):
        names.append(k); cols.append(xs_rank_rows(v, m)[rt, ri])
    def lag(k, skip=0):
        out = np.full((T, N), np.nan); out[k:] = logc[k - skip:T - skip] - logc[:T - k]; return out
    rmean = lambda a, w, mp=None: pd.DataFrame(a).rolling(w, min_periods=mp or w).mean().to_numpy()
    rsum = lambda a, w, mp=None: pd.DataFrame(a).rolling(w, min_periods=mp or w).sum().to_numpy()
    rstd = lambda a, w, mp=None: pd.DataFrame(a).rolling(w, min_periods=mp or w).std().to_numpy()
    for k in [1, 2, 3, 5, 10, 21, 42, 63, 126, 252]: put(f"r{k}", lag(k))
    put("mom12_1", lag(252, 21)); put("mom6_1", lag(126, 21))
    W = 252
    mx = pd.Series(spy).rolling(W, min_periods=200).mean().to_numpy()[:, None]
    my = rmean(dret, W, 200); mxy = rmean(dret * spy[:, None], W, 200)
    vx = pd.Series(spy).rolling(W, min_periods=200).var().to_numpy()[:, None]
    beta = np.clip(np.nan_to_num((mxy - mx * my) / (vx + 1e-12)), -1, 4); del my, mxy
    resid = dret - beta * spy[:, None]
    put("beta", beta)
    for k in [1, 3, 5, 10, 21, 63]: put(f"res{k}", rsum(resid, k))
    put("resmom", rsum(resid, 252, 200) - rsum(resid, 21))
    put("ivol60", rstd(resid, 60, 40)); del resid
    vol5 = rstd(dret, 5); put("vol5", vol5); put("vol20", F["vol20"]); put("vol60", F["vol60"])
    put("volr5_60", vol5 / (F["vol60"] + 1e-9)); del vol5
    put("volr20_120", F["vol20"] / (F["vol120"] + 1e-9))
    dv = P.dollar_vol
    dv1 = dv / (F["adv20"] + 1); put("dv1_adv20", dv1)
    put("adv5_60", rmean(dv, 5, 3) / (F["adv60"] + 1)); put("adv20_60", F["adv20"] / (F["adv60"] + 1))
    put("ladv", np.log(F["adv20"] + 1))
    put("n5_60", rmean(P.n, 5, 3) / (rmean(P.n, 60, 40) + 1)); put("n1_20", P.n / (rmean(P.n, 20, 15) + 1))
    co, oc = F["co"], F["oc"]
    for k in [1, 5, 21]:
        put(f"co{k}", rsum(co, k) if k > 1 else co); put(f"oc{k}", rsum(oc, k) if k > 1 else oc)
    hl = (P.high - P.low) / (P.close + 1e-9); rp = (P.close - P.low) / (P.high - P.low + 1e-9)
    put("rp1", rp); put("rp5", rmean(rp, 5, 3)); put("hl1", hl); put("hl5", rmean(hl, 5, 3)); put("hl21", rmean(hl, 21, 15)); del hl, rp
    vwd = P.close / P.vw - 1; put("vwd1", vwd); put("vwd5", rmean(vwd, 5, 3)); del vwd
    cf = F["closef"]
    put("hi52", cf / F["hi252"]); put("lo52", cf / pd.DataFrame(cf).rolling(252, min_periods=200).min().to_numpy())
    put("ma20", cf / rmean(cf, 20, 15) - 1); put("ma50", cf / F["ma50"] - 1); put("ma200", cf / F["ma200"] - 1)
    put("amihud", rmean(np.abs(dret) / (dv + 1), 21, 15))
    put("max21", pd.DataFrame(dret).rolling(21).max().to_numpy()); put("min21", pd.DataFrame(dret).rolling(21).min().to_numpy())
    put("skew21", pd.DataFrame(dret).rolling(21).skew().to_numpy())
    put("updays10", rsum((dret > 0).astype(float), 10))
    put("r1_x_dv", dret * dv1); put("r5_x_dv", rsum(dret * dv1, 5)); put("abs_r1", np.abs(dret))
    put("lpx", np.log(cf))
    gc.collect()
    return names, np.stack(cols, axis=1)

def fwd_target(P, F, m, h, skip=0):
    """rank-normalised forward log return from close t+skip to close t+skip+h (mlfeat 'cc' when skip=0)."""
    T, N = P.T, P.N
    cum = np.cumsum(np.log1p(np.clip(F["dret"], -0.95, 10)), axis=0)
    L = h + skip; fwd = np.full((T, N), np.nan); fwd[:T - L] = cum[L:] - cum[skip:T - L + skip] if skip else cum[L:] - cum[:T - L]
    has = ~np.isnan(P.close); okk = has.copy(); okk[:T - L] &= has[L:]; okk[T - L:] = False
    df = pd.DataFrame(np.where(m & okk & np.isfinite(fwd), fwd, np.nan))
    r = df.rank(axis=1, pct=True).to_numpy(); cnt = df.notna().sum(axis=1).to_numpy()[:, None]
    return (r - 0.5 - 0.5 / np.maximum(cnt, 1)).astype(np.float32)

def walk_forward(P, m, rows, X, names, y, label_len, retrain=126, start=START, threads=2, rounds=400, leaves=31, lr=0.03, verbose=True):
    """ml_run.py procedure on stacked rows: expanding window from day 261, retrain every `retrain` trading days
    (or 'Y' = first trading day of each year), train rows t with t + label_len <= cut - 1."""
    import lightgbm as lgb
    rt, ri = rows; T = P.T
    yr_ = y[rt, ri]; pred = np.full(len(rt), np.nan, np.float32)
    i0 = int(np.searchsorted(P.dates, pd.Timestamp(start)))
    if retrain == "Y":
        cuts = sorted(set([i0] + [int(np.searchsorted(P.dates, pd.Timestamp(f"{y_}-01-01"))) for y_ in range(P.dates[i0].year + 1, P.dates[-1].year + 1)]))
    else:
        cuts = list(range(i0, T, retrain))
    cuts = cuts + [T]
    params = dict(objective="regression", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=1000, feature_fraction=0.6,
                  bagging_fraction=0.6, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=threads, max_bin=63, seed=0)
    imps = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        last_tr = a - label_len - 1
        tr = (rt >= 261) & (rt <= last_tr) & np.isfinite(yr_); te = (rt >= a) & (rt < b)
        if last_tr - 261 < 252: continue
        t1 = time.time()
        mdl = lgb.train(params, lgb.Dataset(X[tr], yr_[tr], feature_name=names, free_raw_data=True), num_boost_round=rounds)
        pred[te] = mdl.predict(X[te], num_threads=threads)
        imp = pd.Series(mdl.feature_importance("gain"), index=names); imps.append(imp / imp.sum())
        if verbose: log(f"   fit {P.dates[a].date()}..{P.dates[b-1].date()} train rows {tr.sum()} {time.time()-t1:.0f}s")
    out = np.full((T, P.N), np.nan, np.float32); out[rt, ri] = pred
    return out, pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)

def rank_ic(pred, y, m, i0, i1):
    ic = []
    for d in range(i0, i1):
        ok = m[d] & np.isfinite(pred[d]) & np.isfinite(y[d])
        ic.append(pd.Series(pred[d, ok]).corr(pd.Series(y[d, ok]), method="spearman") if ok.sum() > 50 else np.nan)
    return pd.Series(ic)

def save_pred(a, name): np.savez_compressed(f"{OUT}/pred_{name}.npz", p=a.astype(np.float16))
def load_pred(name): return np.load(f"{OUT}/pred_{name}.npz")["p"].astype(np.float32)

# ------------------------------------------------------------------ liquidity matrices for the cost model
def liq_mats(P):
    """T x N lagged closing-auction dollar volume proxy max(exp(dvc), 0.05 exp(ldv20)) from liq.raw_frame
    (2000-name universe; NaN elsewhere -> flat 3 bps), and the daily vol rv20 (20d sd of close-to-close, lagged 1 day)."""
    cp = f"{OUT}/liqmat.npz"
    if os.path.exists(cp):
        z = np.load(cp);
        if z["aud"].shape == (P.T, P.N): return z["aud"].astype(np.float64)
    import liq
    aud = np.full((P.T, P.N), np.nan, np.float32); n = 0
    for tk in P.tickers:
        if tk not in rlib.RANK or not os.path.exists(f"{rlib.DATA}/cache_liq/{tk}.parquet"): continue
        d = pd.read_parquet(f"{rlib.DATA}/cache_liq/{tk}.parquet", columns=["ldv20", "dvc"])
        d = d[d.index <= DEV_END].reindex(P.dates)
        aud[:, P.ix[tk]] = np.maximum(np.exp(d.dvc.to_numpy()), 0.05 * np.exp(d.ldv20.to_numpy())); n += 1
    log(f"liq_mats: {n} names with liq data")
    np.savez_compressed(cp, aud=aud); return aud.astype(np.float64)

def auction_bps(order_usd, aud, rvd, c=0.3):
    """per-side bps: max(1, c * rv * 1e4 * sqrt(order/aud)); 3 bps where aud is unknown."""
    with np.errstate(all="ignore"):
        x = np.maximum(1.0, c * rvd * 1e4 * np.sqrt(order_usd / np.maximum(aud, 1.0)))
    return np.where(np.isfinite(aud) & np.isfinite(rvd), x, 3.0)

# ------------------------------------------------------------------ portfolio simulator
def beta_mat(F):
    dret = F["dret"]; spy = F["spy_ret"]; W = 252
    rm = lambda a: pd.DataFrame(a).rolling(W, min_periods=200).mean().to_numpy()
    mx = pd.Series(spy).rolling(W, min_periods=200).mean().to_numpy()[:, None]
    vx = pd.Series(spy).rolling(W, min_periods=200).var().to_numpy()[:, None]
    return np.clip(np.nan_to_num((rm(dret * spy[:, None]) - mx * rm(dret)) / (vx + 1e-12)), -1, 4)

class Sim:
    """Holdings-based simulator: holdings drift with returns; a name is traded at the close only when its target
    changes by more than `mt` (fraction of equity) or it enters/exits; orders = names traded that day (+SPY)."""
    def __init__(self, P, F, m, aud):
        self.P, self.F, self.m = P, F, m
        self.R = np.nan_to_num(F["dret"]); self.R = np.clip(self.R, -0.95, 3.0)
        self.aud = aud; vol20 = np.vstack([np.full((1, P.N), np.nan), F["vol20"][:-1]])
        self.rvd = np.nan_to_num(vol20, nan=0.03)
        self.beta = np.vstack([np.zeros((1, P.N)), beta_mat(F)[:-1]])     # beta known at close t-1 (used for the trade at t)
        self.spy = P.col("SPY")
        self.i0 = int(np.searchsorted(P.dates, pd.Timestamp(START)))

    def targets(self, S, k=25, mode="tranche", hold=5, reb=1, buf=0, timing="A"):
        """yield (t, target weight row) for trade days.  S: T x N score (NaN outside universe).
        mode 'tranche': daily top-k/bottom-k target, averaged over the last `hold` days (agents/ml).
        mode 'band'   : every `reb` days, keep holdings whose rank is within top k+buf (bottom k+buf), fill to k."""
        P = self.P; N = P.N; lag = 1 if timing == "A" else 0
        tq = []; prevL = np.zeros(0, int); prevS = np.zeros(0, int)
        for t in range(self.i0 - hold - 15, P.T):
            s = S[t - lag]
            idx = np.where(np.isfinite(s))[0]
            if mode == "tranche":
                w = np.zeros(N)
                if len(idx) >= 4 * k:
                    o = idx[np.argsort(-s[idx], kind="stable")]; w[o[:k]] = 1.0 / k; w[o[-k:]] = -1.0 / k
                tq.append(w); tq = tq[-hold:]
                if (t - self.i0) % reb == 0 or reb == 1: yield t, np.mean(tq, axis=0)
            else:
                if (t - self.i0) % reb != 0: continue
                if len(idx) < 4 * k: continue
                o = idx[np.argsort(-s[idx], kind="stable")]; rank = np.full(N, -1); rank[o] = np.arange(len(o)); n = len(o)
                keepL = [c for c in prevL if rank[c] >= 0 and rank[c] < k + buf]
                keepS = [c for c in prevS if rank[c] >= 0 and rank[c] >= n - k - buf]
                sl = set(keepL) | set(keepS)
                L = keepL + [c for c in o if c not in sl][: max(0, k - len(keepL))]
                sl |= set(L)
                Sx = keepS + [c for c in o[::-1] if c not in sl][: max(0, k - len(keepS))]
                prevL, prevS = np.array(L, int), np.array(Sx, int)
                w = np.zeros(N); w[L] = 1.0 / k; w[Sx] = -1.0 / k
                yield t, w

    def run(self, S, k=25, mode="tranche", hold=5, reb=1, buf=0, mt=None, hedge=True, hedge_band=0.02, timing="A",
            c=0.3, cap=CAP, order_usd=ORDER_USD, borrow=0.01, legs=False):
        P = self.P; N = P.N; T = P.T; spy = self.spy
        if mt is None: mt = 0.5 / (k * hold) if mode == "tranche" else 0.5 / k
        h = np.zeros(N); gross = np.zeros(T); imp = np.zeros(T); com = np.zeros(T); bor = np.zeros(T); nord = np.zeros(T); turn = np.zeros(T)
        expo = np.zeros(T); bet = np.zeros(T)
        tg = dict(self.targets(S, k, mode, hold, reb, buf, timing))
        for t in range(self.i0 - hold - 15, T):
            # PnL of holdings set at close t-1, realised on day t
            if t > 0:
                r = self.R[t]; g = float(h @ r); gross[t] = g
                h = h * (1 + r) / (1 + g)
            if t in tg:
                w = tg[t].copy()
                d = w - h; d[spy] = 0.0
                tr = (np.abs(d) > mt) | ((w == 0) & (h != 0) & (np.arange(N) != spy)) | ((w != 0) & (h == 0))
                tr[spy] = False
                newh = np.where(tr, w, h); newh[spy] = h[spy]
                if hedge:
                    bt_ = float(np.delete(newh, spy) @ np.delete(self.beta[t], spy)); tgt = -np.clip(bt_, -0.5, 0.5)
                    if abs(newh[spy] - tgt) > hedge_band: newh[spy] = tgt
                elif newh[spy] != 0: newh[spy] = 0.0
                dd = np.abs(newh - h); trd = dd > 1e-12
                if trd.any():
                    cb = auction_bps(dd[trd] * cap, self.aud[t, trd], self.rvd[t, trd], c)
                    imp[t] = float(dd[trd] @ cb) / 1e4; nord[t] = trd.sum(); com[t] = nord[t] * order_usd / cap; turn[t] = dd.sum()
                h = newh
            bor[t] = np.clip(-h, 0, None).sum() * borrow / 252
            expo[t] = np.abs(h).sum(); bet[t] = float(h @ self.beta[t])
        # costs at close t charged on t; borrow on positions held over night t->t+1 charged on t+1
        bor = np.r_[0.0, bor[:-1]]
        sl = slice(self.i0, T); ix = P.dates[sl]
        df = pd.DataFrame({"gross": gross[sl], "imp": imp[sl], "com": com[sl], "bor": bor[sl], "orders": nord[sl], "turn": turn[sl], "expo": expo[sl], "beta": bet[sl]}, index=ix)
        df["net"] = df.gross - df.imp - df.com - df.bor
        return df

def summ(df, name=""):
    n = df.net; s = rlib.sh(n); eq = (1 + n).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    h1 = n[n.index < "2021-10-28"]; h2 = n[n.index >= "2021-10-28"]
    oos, _ = rlib.fold_oos(n, p=False)
    return dict(name=name, sh=round(s, 2), foos=round(rlib.sh(oos), 2), gross=round(rlib.sh(df.gross), 2), ret=round(n.mean() * 252 * 100, 1),
                vol=round(n.std() * np.sqrt(252) * 100, 1), mdd=round(mdd * 100, 1), h1=round(rlib.sh(h1), 2), h2=round(rlib.sh(h2), 2),
                ord_d=round(df.orders.mean(), 1), turn_d=round(df.turn.mean(), 3), imp_bp=round(df.imp.mean() * 1e4, 2), com_bp=round(df.com.mean() * 1e4, 2),
                bor_bp=round(df.bor.mean() * 1e4, 2), gross_bp=round(df.gross.mean() * 1e4, 2), beta=round(df.beta.mean(), 2))
