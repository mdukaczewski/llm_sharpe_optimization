"""etfmacro lane helpers: daily ETF data (vlib) cut at DEV_END, FRED VIX cut, minute-bar ETF frames."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/agents/vol")
import rlib
from rlib import DEV_END, START, sh, stats, fold_oos, save_cand
import vlib
LANE = "etfmacro"

def load_etf():
    """vlib.load() cut at DEV_END (all frames)."""
    D = vlib.load()
    return {k: v.loc[:DEV_END] for k, v in D.items()}

def load_fred(idx=None):
    F = pd.read_csv(f"{ROOT}/agents/vol/fred_vix.csv", index_col=0, parse_dates=True).loc[:DEV_END]
    if idx is not None: F = F.reindex(idx).ffill()
    return F

def bt(W, D, cost_bps=1.5, fill="close", lag=1):
    """vlib.backtest cut at DEV_END; W decided at close t, filled at close t+1 (lag=1)."""
    W = W.loc[:DEV_END]
    return vlib.backtest(W, D["ret"], cost_bps=cost_bps, fill=fill, D=D, start=START, end=DEV_END, lag=lag)

def tf(tk):
    """rlib.ticker_frame cut at DEV_END (targets at DEV_END itself are NaN-safe: y_on of DEV_END uses 04-28 -> drop)."""
    d = rlib.ticker_frame(tk).loc[START:DEV_END].copy()
    return d

def panel():
    import bt as btmod
    return btmod.Panel(f"{ROOT}/data/panel_liquid.npz")

def pseries(P, tk, field="close"):
    s = pd.Series(getattr(P, field)[:, P.col(tk)], index=P.dates); return s.loc[:DEV_END]

def log(msg):
    with open(f"{HERE}/log.txt", "a") as fh: fh.write(f"{pd.Timestamp.now():%H:%M:%S} {msg}\n")
    print(msg, flush=True)

def corr_report(r, p=True):
    """corr with sleeves/vol_ens2_net.csv, sleeves/onlgb1200q_ls20.csv and SPY close-to-close."""
    out = {}
    for nm, f in [("vol_ens2", "vol_ens2_net.csv"), ("onlgb1200q", "onlgb1200q_ls20.csv")]:
        s = pd.read_csv(f"{ROOT}/sleeves/{f}", index_col=0, parse_dates=True).iloc[:, 0].loc[:DEV_END]
        out[nm] = float(r.reindex(s.index).fillna(0).corr(s))
    spy = load_etf()["ret"]["SPY"]
    out["SPY"] = float(r.reindex(spy.index).fillna(0).corr(spy))
    if p: print({k: round(v, 3) for k, v in out.items()})
    return out

# ---------------------------------------------------------------- overnight ETP legs from minute bars (split-safe, no |y| cap)
_MIN = {}
def minute(tk):
    from ilib import Min
    if tk not in _MIN: _MIN[tk] = Min(tk, end=DEV_END)
    return _MIN[tk]

def on_ret(tk, entry_bin=390, exit_bin=1):
    """Overnight return indexed at the DECISION day t: fill at close of `entry_bin` of day t (390 = official close,
    i.e. a decision at bin 389 / the close) -> fill at close of `exit_bin` of day t+1. Split-adjusted (Min.sf)."""
    M = minute(tk); pin = M.c[:, entry_bin]; pout = np.r_[M.c[1:, exit_bin], np.nan]; sf = np.r_[M.sf[1:], 1.0]
    r = pout / (pin * sf) - 1
    return pd.Series(r, index=pd.DatetimeIndex(M.dates))

def leg_ret(w, r, cost=1.5, borrow=0.0):
    """w: weights decided at t (Series), r: returns of the leg indexed at the decision day t. Net return booked at t+1 (the exit day).
    cost bps per side on |w| (2 sides), borrow annual on |short|."""
    idx = w.index; r = r.reindex(idx)
    g = w * r; c = w.abs() * 2 * cost / 1e4 + w.clip(upper=0).abs() * borrow / 252
    n = (g - c); n[r.isna()] = 0.0
    return n.shift(1).fillna(0.0).loc[START:DEV_END]
