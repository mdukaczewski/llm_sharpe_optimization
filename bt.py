#!/usr/bin/env python3
"""Backtest library: loads panel.npz, builds liquid universe, runs
periodic-rebalance long-only (or long/cash) portfolios with costs."""
import numpy as np, pandas as pd, sys, os

PANEL = os.environ.get("PANEL", os.path.expanduser("~/strat20/data/panel.npz"))
if not os.path.exists(PANEL): PANEL = os.path.expanduser("~/strat20/data/panel_liquid.npz")   # local box: the slim panel is a superset of every universe screen

class Panel:
    def __init__(self, path=PANEL, start="2014-06-01"):
        z = np.load(path, allow_pickle=True)
        self.dates = pd.to_datetime(z["dates"])
        self.tickers = z["tickers"]
        self.ttype = z["ttype"]
        adj = z["adj"].astype(np.float64)
        self.close_raw = z["close"].astype(np.float64)
        self.close = self.close_raw * adj
        self.open = z["open"].astype(np.float64) * adj
        self.high = z["high"].astype(np.float64) * adj
        self.low = z["low"].astype(np.float64) * adj
        self.vw = z["vw"].astype(np.float64) * adj
        self.vol = z["vol"].astype(np.float64) / adj
        self.n = z["n"].astype(np.float64)
        self.div = z["div"].astype(np.float64) * adj
        self.T, self.N = self.close.shape
        # total-return close-to-close returns
        prev = np.vstack([np.full((1, self.N), np.nan), self.close[:-1]])
        self.ret = (self.close + self.div) / prev - 1.0
        self.dollar_vol = self.close * self.vol
        self.ix = {t: i for i, t in enumerate(self.tickers)}

    def col(self, t):
        return self.ix[t]

def ffill(a, limit=None):
    df = pd.DataFrame(a)
    return df.ffill(limit=limit).values

def rolling_sum(a, w):
    df = pd.DataFrame(a)
    return df.rolling(w, min_periods=w).sum().values

def rolling_mean(a, w, minp=None):
    df = pd.DataFrame(a)
    return df.rolling(w, min_periods=minp or w).mean().values

def rolling_std(a, w, minp=None):
    df = pd.DataFrame(a)
    return df.rolling(w, min_periods=minp or w).std().values

def rolling_max(a, w, minp=None):
    df = pd.DataFrame(a)
    return df.rolling(w, min_periods=minp or w).max().values

def month_ends(dates):
    """indices of last trading day of each month"""
    s = pd.Series(np.arange(len(dates)), index=dates)
    return s.groupby([dates.year, dates.month]).last().values

def week_ends(dates):
    s = pd.Series(np.arange(len(dates)), index=dates)
    iso = dates.isocalendar()
    return s.groupby([iso.year.values, iso.week.values]).last().values

def run(panel, weights_fn, rebal_idx, start, end, cost_bps=10.0, verbose=False):
    """weights_fn(i) -> dict ticker_col -> weight (sum<=1), decided at close of
    day i using data up to i. Positions are entered at the NEXT day's open? No:
    we trade at close of day i (signal from close i) -- conservative variant
    below uses next-day open execution: weights apply from day i+1's close
    return, and the day i+1 return is measured open(i+1)->close(i+1) for new
    positions.  To keep it simple and conservative we execute at next day's
    CLOSE (signal at close i, trade at close i+1, earn from i+2). Cash earns 0.
    """
    P = panel
    T, N = P.T, P.N
    i0 = int(np.searchsorted(P.dates, pd.Timestamp(start)))
    i1 = int(np.searchsorted(P.dates, pd.Timestamp(end), side="right"))
    w = np.zeros(N)                 # current weights (fraction of equity)
    equity = 1.0
    eq = np.full(T, np.nan)
    turnover_total = 0.0
    rebal_set = set(int(i) for i in rebal_idx)
    pending = None
    ntrades = 0
    hold_counts = []
    for i in range(i0, i1):
        # apply day i return to holdings (weights drift)
        r = P.ret[i]
        r = np.where(np.isnan(r), 0.0, r)
        # delisted/stale: if no close for 10+ days we exit at last close (0 ret)
        gross = w * (1 + r)
        equity *= (1 + np.sum(w * r))
        tot = gross.sum()
        w = gross / equity if equity > 0 else gross * 0
        # w now = value fraction after the day's move... normalize by equity:
        # equity already includes gains; gross sums position values relative
        # to pre-day equity; divide by (1+port_ret)
        # (that is what the division above does since equity updated)
        # NOTE: w computed as gross/equity assumes equity started at 1 at this
        # step -- fix by tracking relative:
        # (we recompute properly below)
        if pending is not None:
            # execute the trade at this close
            new_w = pending
            turnover = np.abs(new_w - w).sum()
            equity *= (1 - cost_bps / 1e4 * turnover)
            turnover_total += turnover
            ntrades += int((np.abs(new_w - w) > 1e-9).sum())
            w = new_w
            pending = None
            hold_counts.append(int((w > 0).sum()))
        if i in rebal_set:
            tw = weights_fn(i)
            new_w = np.zeros(N)
            for c, x in tw.items():
                new_w[c] = x
            pending = new_w
        eq[i] = equity
    s = pd.Series(eq[i0:i1], index=P.dates[i0:i1])
    return s, dict(turnover=turnover_total, ntrades=ntrades, avg_hold=np.mean(hold_counts) if hold_counts else 0)

def run_fast(panel, weights_fn, rebal_idx, start, end, cost_bps=10.0, exec_lag=1, cash_rate=0.0):
    """Correct drift accounting. Signal at close of rebalance day i, executed
    at close of day i+exec_lag. Daily portfolio return = sum(w * r)."""
    P = panel
    T, N = P.T, P.N
    i0 = int(np.searchsorted(P.dates, pd.Timestamp(start)))
    i1 = int(np.searchsorted(P.dates, pd.Timestamp(end), side="right"))
    w = np.zeros(N)
    equity = 1.0
    eq = np.full(T, np.nan)
    tto = 0.0; nreb = 0; holds = []
    pend = {}
    rebal_set = set(int(i) for i in rebal_idx)
    nan_days = np.zeros(N, dtype=int)
    for i in range(i0, i1):
        r = P.ret[i]
        miss = np.isnan(r)
        r = np.where(miss, 0.0, r)
        pr = float(np.dot(w, r))
        equity *= (1 + pr + (1 - w.sum()) * cash_rate / 252)
        w = w * (1 + r) / (1 + pr)
        # positions with no bar for 15 straight days are considered delisted
        # and liquidated at last price with an extra 50% haircut? -> use 0 ret
        nan_days = np.where(miss, nan_days + 1, 0)
        dead = (w > 0) & (nan_days >= 15)
        if dead.any():
            w[dead] = 0.0
        if i in pend:
            new_w = pend.pop(i)
            turnover = float(np.abs(new_w - w).sum())
            equity *= (1 - cost_bps / 1e4 * turnover)
            tto += turnover; nreb += 1
            w = new_w * (1 - cost_bps / 1e4 * turnover)   # keep sums consistent
            holds.append(int((w > 0).sum()))
        if i in rebal_set:
            tw = weights_fn(i)
            new_w = np.zeros(N)
            for c, x in tw.items():
                new_w[c] = x
            pend[i + exec_lag] = new_w
        eq[i] = equity
    s = pd.Series(eq[i0:i1], index=P.dates[i0:i1])
    return s, dict(turnover=tto, nrebal=nreb, avg_hold=float(np.mean(holds)) if holds else 0.0)

def stats(eq, name="", rf=0.0):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - rf) / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    # first-year partial
    yrets = eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    return dict(name=name, cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd, years=yrs, total=eq.iloc[-1] / eq.iloc[0] - 1, yearly=yrets)

def report(eq, name="", extra=None):
    s = stats(eq, name)
    line = f"{name:40s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:4.2f}  MaxDD {s['maxdd']*100:6.1f}%  total x{1+s['total']:.2f}"
    if extra:
        line += "  " + " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in extra.items())
    print(line, flush=True)
    print("   yearly: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in s["yearly"].items()), flush=True)
    return s
