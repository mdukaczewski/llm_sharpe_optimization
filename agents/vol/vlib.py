"""Small daily backtest engine for ETF/ETP time-series strategies.
Weights W[t] (dict or DataFrame, columns=tickers) are decided at the CLOSE of day t
and FILLED AT THE NEXT DAY'S CLOSE (t+1); they earn close-to-close returns from t+2 on.
Daily rebalance to target with drift-adjusted turnover. Costs: cost_bps per unit of
one-way turnover + borrow (annual rate / 252 on |short weight|). Cash earns 0."""
import numpy as np, pandas as pd
import os as _os
CSV = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "etf.csv")
START, END = "2016-10-28", "2026-10-27"
BORROW = {"VIXY": 0.02, "VXX": 0.02, "UVXY": 0.05, "TQQQ": 0.05, "SQQQ": 0.05, "SVXY": 0.05,
          "SPXL": 0.05, "SPXS": 0.05, "UPRO": 0.05, "SPXU": 0.05, "SSO": 0.05, "SDS": 0.05,
          "QLD": 0.05, "QID": 0.05, "TNA": 0.05, "TZA": 0.05, "SOXL": 0.05, "SOXS": 0.05,
          "VIXM": 0.02, "VXZ": 0.02, "UVIX": 0.05, "SVIX": 0.05}
DEFAULT_BORROW = 0.01   # SPY/QQQ/TLT etc.

def load():
    df = pd.read_csv(CSV, index_col=0, parse_dates=True)
    tks = sorted({c.rsplit("_", 1)[0] for c in df.columns})
    D = {}
    for k in ["open", "high", "low", "close", "vw"]:
        D[k] = pd.DataFrame({t: df[f"{t}_{k}"] * df[f"{t}_adj"] for t in tks})
    D["div"] = pd.DataFrame({t: df[f"{t}_div"] * df[f"{t}_adj"] for t in tks}).fillna(0)
    D["vol"] = pd.DataFrame({t: df[f"{t}_vol"] / df[f"{t}_adj"] for t in tks})
    c = D["close"]
    D["ret"] = (c + D["div"]) / c.shift(1) - 1          # total return close->close
    D["oc"] = c / D["open"] - 1                         # intraday
    D["co"] = (D["open"] + D["div"]) / c.shift(1) - 1   # overnight
    return D

def backtest(W, ret, cost_bps=2.0, borrow=BORROW, start=START, end=END, lag=1, fill="close", D=None):
    """W: DataFrame (dates x tickers) target weights decided at close t.
    fill='close': filled at close t+lag (default lag=1), earns from the following day.
    fill='open' : filled at the OPEN of t+1 (needs D with 'co','oc'): day t+1 return =
                  (1 + held.co)(1 + tgt.oc) - 1, i.e. old book earns overnight, new book earns intraday.
    Returns DataFrame with gross/net daily returns, turnover and borrow cost."""
    W = W.reindex(ret.index).fillna(0.0)
    R = ret[W.columns].reindex(W.index).fillna(0.0).values
    Wv = W.values
    T, K = Wv.shape
    br = np.array([borrow.get(t, DEFAULT_BORROW) for t in W.columns]) / 252.0
    i0 = int(np.searchsorted(W.index, pd.Timestamp(start))); i1 = int(np.searchsorted(W.index, pd.Timestamp(end), side="right"))
    held = np.zeros(K)       # weights held after close of day i-1
    gross = np.zeros(T); net = np.zeros(T); turn = np.zeros(T); bcost = np.zeros(T)
    if fill == "open":
        CO = D["co"][W.columns].reindex(W.index).fillna(0.0).values; OC = D["oc"][W.columns].reindex(W.index).fillna(0.0).values
    for i in range(i0, i1):
        b = float(np.sum(np.abs(np.minimum(held, 0)) * br))
        if fill == "open":
            tgt = Wv[i - 1] if i >= 1 else held
            r1 = float(held @ CO[i]); drift = held * (1 + CO[i]) / (1 + r1)
            to = float(np.abs(tgt - drift).sum())
            r2 = float(tgt @ OC[i])
            pr = (1 + r1) * (1 + r2) - 1
            held = tgt * (1 + OC[i]) / (1 + r2)
        else:
            r = R[i]
            pr = float(held @ r)
            drift = held * (1 + r) / (1 + pr) if abs(1 + pr) > 1e-9 else held * 0
            j = i - lag                    # signal day whose target is filled today
            tgt = Wv[j] if j >= 0 else drift
            to = float(np.abs(tgt - drift).sum())
            held = tgt
        gross[i] = pr; turn[i] = to; bcost[i] = b
        net[i] = pr - to * cost_bps / 1e4 - b
    idx = W.index[i0:i1]
    return pd.DataFrame({"gross": gross[i0:i1], "net": net[i0:i1], "turn": turn[i0:i1], "borrow": bcost[i0:i1]}, index=idx)

def sh(r):
    s = r.std()
    return r.mean() / s * np.sqrt(252) if s > 0 else np.nan

def window(r, a, b):
    x = r.loc[a:b]; return float(np.exp(np.log1p(x).sum()) - 1)

def stats(bt, name="", full=True, quiet=False):
    g, n = bt["gross"], bt["net"]
    eq = np.exp(np.log1p(n).cumsum()); dd = (eq / eq.cummax() - 1).min()
    yr = n.groupby(n.index.year).apply(sh)
    wk = n.rolling(5).apply(lambda x: np.exp(np.log1p(x).sum()) - 1)
    out = dict(name=name, sharpe_g=sh(g), sharpe_n=sh(n), ret=n.mean() * 252, vol=n.std() * np.sqrt(252), cagr=eq.iloc[-1] ** (252 / len(n)) - 1,
               maxdd=dd, worst_day=n.min(), worst_day_dt=n.idxmin(), worst_wk=wk.min(), worst_wk_dt=wk.idxmin(), turn_yr=bt["turn"].sum() / len(n) * 252,
               feb18=window(n, "2018-02-01", "2018-02-15"), mar20=window(n, "2020-02-20", "2020-03-31"), aug24=window(n, "2024-07-31", "2024-08-09"),
               worst_yr=yr.min(), worst_yr_y=yr.idxmin(), yr=yr)
    if not quiet:
        print(f"{name:48s} Sh g/n {out['sharpe_g']:5.2f}/{out['sharpe_n']:5.2f} ret {out['ret']*100:6.1f}% vol {out['vol']*100:5.1f}% MDD {dd*100:6.1f}% "
              f"wd {out['worst_day']*100:6.1f}% ({out['worst_day_dt'].date()}) wwk {out['worst_wk']*100:6.1f}% turn/yr {out['turn_yr']:5.1f} "
              f"Feb18 {out['feb18']*100:+5.1f}% Mar20 {out['mar20']*100:+5.1f}% Aug24 {out['aug24']*100:+5.1f}% worstyr {out['worst_yr']:5.2f}({out['worst_yr_y']})", flush=True)
        if full:
            print("      yearly Sharpe: " + " ".join(f"{y}:{v:+.2f}" for y, v in yr.items()), flush=True)
    return out

def rv(ret, w):
    return ret.rolling(w).std() * np.sqrt(252)
