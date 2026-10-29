"""Intraday research lib. Minute bins k=0..390 <-> 09:30..16:00 NY (bin 390 = 16:00
bar = closing auction print). Conventions:
 * a trade decided at bin m fills at close of bin m+1 (never the signal bar)
 * exit 'at the close' = decided at bin 389, filled at c[390]
 * cost per side (bps): SPY/QQQ 0.5, other ETFs 1.5, mega-cap stocks 1.0
 * Sharpe = mean/std*sqrt(252) of daily returns, rf=0 (same as ls.sh)
"""
import numpy as np, pandas as pd, json, os
NPZ = "/home/trumpet/strat20/data/minute_npz"
START, END = "2016-10-28", "2026-10-27"
COST = {"SPY": 0.5, "QQQ": 0.5, "AAPL": 1.0, "MSFT": 1.0, "NVDA": 1.0, "AMZN": 1.0, "TSLA": 1.0}
def cost_of(tk): return COST.get(tk, 1.5)
ETF4 = ["SPY", "QQQ", "IWM", "DIA"]
ALL = ["SPY","QQQ","IWM","DIA","TLT","GLD","XLF","XLK","XLE","AAPL","MSFT","NVDA","AMZN","TSLA","TQQQ","SVXY","UVXY","VIXY"]

_splits = None
def split_factor(tk, dates):
    """multiplicative factor to apply to PRIOR-day prices so they compare with day d:
    returns array f[d] = price ratio new/old applied on day d (1 except split days)."""
    global _splits
    if _splits is None:
        _splits = json.load(open("/home/trumpet/strat20/data/splits.json"))
    f = np.ones(len(dates))
    for s in _splits:
        if s["ticker"] != tk: continue
        ed = pd.Timestamp(s["execution_date"])
        i = np.searchsorted(dates, ed)
        if 0 <= i < len(dates) and dates[i] == ed:
            f[i] = s["split_from"] / s["split_to"]   # prev close must be multiplied by this
    return f

_panel = None
def official_close(tk, dates):
    global _panel
    if _panel is None:
        z = np.load("/home/trumpet/strat20/data/panel_liquid.npz", allow_pickle=True)
        _panel = (pd.to_datetime(z["dates"]), list(z["tickers"]), z["close"].astype(np.float64))
    pd_, tks, cl = _panel
    if tk not in tks: return np.full(len(dates), np.nan)
    s = pd.Series(cl[:, tks.index(tk)], index=pd_)
    return s.reindex(dates).values

class Min:
    def __init__(self, tk, start=START, end=END, ffill=True):
        z = np.load(f"{NPZ}/{tk}.npz")
        d = pd.to_datetime(z["dates"].astype(str))
        m = (d >= pd.Timestamp(start) - pd.Timedelta(days=40)) & (d <= pd.Timestamp(end))
        self.tk = tk
        self.dates = d[m]
        self.nbars = z["nbars"][m]
        for k in ["o", "h", "l", "c", "vw"]:
            a = z[k][m].astype(np.float64)
            setattr(self, k + "_raw", a.copy())
            if ffill:
                a = pd.DataFrame(a.T).ffill().values.T      # ffill forward within day
            setattr(self, k, a)
        self.v = z["vol"][m].astype(np.float64)
        self.n = z["n"][m].astype(np.float64)
        # MOC fill = official close from the daily panel (16:00 bar can contain post-close prints)
        oc = official_close(tk, self.dates)
        self.c[:, 390] = np.where(np.isfinite(oc), oc, self.c[:, 389])
        self.close = self.c[:, 390]
        self.open = self.o_raw[:, 0]
        bad = ~np.isfinite(self.open)
        self.open[bad] = self.c[bad, 0]
        self.sf = split_factor(tk, self.dates)
        self.prev_close = np.r_[np.nan, self.close[:-1]] * self.sf
        self.gap = self.open / self.prev_close - 1
        self.cc = self.close / self.prev_close - 1
        self.oc = self.close / self.open - 1
        self.half = self.nbars < 300                         # half days
        self.insample = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        self.cost = cost_of(tk)
        # cumulative vwap by minute
        pv = np.nan_to_num(self.vw_raw) * self.v
        self.cvwap = np.cumsum(pv, 1) / np.maximum(np.cumsum(self.v, 1), 1e-9)

    def r(self, a, b):
        """return from close of bin a to close of bin b (a<b); a=-1 means the open"""
        pa = self.open if a < 0 else self.c[:, a]
        return self.c[:, b] / pa - 1

def fill(M, m):
    """fill price for a decision taken at end of bin m: close of bin m+1 (cap at 390)"""
    return M.c[:, min(m + 1, 390)]

def leg(M, side, m_in, m_out, cost_bps=None):
    """daily gross/net return of holding `side` (array, -1/0/+1 or sized) from the fill
    after decision at m_in until the fill after decision at m_out. side given per day."""
    cost = M.cost if cost_bps is None else cost_bps
    pin, pout = fill(M, m_in), fill(M, m_out)
    g = side * (pout / pin - 1)
    n = g - np.abs(side) * 2 * cost / 1e4
    return g, n

def sh(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)

def stats(dates, g, n, side=None, name="", mask=None, p=True):
    g = pd.Series(g, index=dates); n = pd.Series(n, index=dates)
    if mask is not None:
        g, n = g[mask], n[mask]
    g, n = g.fillna(0), n.fillna(0)
    eq = (1 + n).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yr = n.groupby(n.index.year).apply(sh)
    trades = 0.0
    if side is not None:
        s = pd.Series(side, index=dates)
        if mask is not None: s = s[mask]
        trades = (np.abs(s) > 0).sum() / (len(s) / 252)
    d = dict(name=name, sh_g=sh(g), sh_n=sh(n), ret=n.mean() * 252, vol=n.std() * np.sqrt(252), mdd=dd,
             trades=trades, yr=yr, n=n, g=g)
    if p:
        print(f"{name:42s} Sh gross {d['sh_g']:5.2f} net {d['sh_n']:5.2f} | ret {d['ret']*100:6.2f}% vol {d['vol']*100:5.2f}% mdd {dd*100:6.1f}% | trades/yr {trades:5.0f} | yrs " +
              " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()), flush=True)
    return d

def combine(rets, lookback=60, p=True, name="COMBO", vol_target=0.10):
    """equal-vol-weight legs (trailing realized vol, lagged) -> daily portfolio return"""
    df = pd.concat(rets, axis=1).fillna(0)
    v = df.rolling(lookback, min_periods=20).std().shift(1).fillna(method="bfill") * np.sqrt(252)
    w = (vol_target / len(df.columns)) / v.clip(lower=0.02)
    port = (df * w).sum(1)
    return port

def pstats(port, name="COMBO", gross=None):
    n = port.fillna(0); eq = (1 + n).cumprod(); dd = (eq / eq.cummax() - 1).min()
    yr = n.groupby(n.index.year).apply(sh)
    gs = f"gross {sh(gross):5.2f} " if gross is not None else ""
    print(f"{name:42s} Sh {gs}net {sh(n):5.2f} | ret {n.mean()*252*100:6.2f}% vol {n.std()*np.sqrt(252)*100:5.2f}% mdd {dd*100:6.1f}% | yrs " +
          " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()), flush=True)

# ---------- position-path engine ----------
def path_returns(M):
    """r[d,k] = return over interval k: k<390: c[k]->c[k+1]; k=390: c[390](d) -> c[0](d+1) (split adj)."""
    c = M.c
    r = np.zeros((len(M.dates), 391))
    r[:, :390] = c[:, 1:] / c[:, :390] - 1
    r[:-1, 390] = c[1:, 0] / (c[:-1, 390] * M.sf[1:]) - 1
    return np.nan_to_num(r)

def pathbt(M, pos, cost_bps=None):
    """pos[d,k] = position held over interval k (see path_returns). A decision at bin m
    may first affect pos[d, m+1]. Costs charged on every |change| of position, including
    overnight carry-over. Returns (gross, net, trades_per_day) as daily arrays."""
    cost = (M.cost if cost_bps is None else cost_bps) / 1e4
    r = path_returns(M)
    pos = np.nan_to_num(pos)
    g = (pos * r).sum(1)
    prev = np.r_[0.0, pos[:-1, 390]]
    dp = np.abs(np.diff(np.c_[prev, pos], axis=1))
    n = g - dp.sum(1) * cost
    return g, n, (dp > 1e-12).sum(1)

def zs(x, w=60):
    s = pd.Series(x); return (s / s.rolling(w, min_periods=20).std().shift(1)).values
