"""etfon lane helpers: per-ETF close-time features + overnight targets (official open and 09:31 bar), costs, reporting."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import rlib, liq
from rlib import DEV_END, START, sh, stats, fold_oos
LANE = "etfon"
EQ = ["SPY", "QQQ", "IWM", "DIA", "EEM", "EFA", "KRE", "SMH", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
OTH = ["GDX", "GLD", "HYG", "IEF", "SLV", "TLT", "USO"]
ALL = EQ + OTH            # TQQQ excluded from the risk-weighted books (3x QQQ duplicate); used only as a check
COMM_BPS = 3.0 / 400000 * 1e4      # $3 per order on $0.4M = 0.075 bp of equity per order

def log(msg):
    with open(f"{HERE}/log.txt", "a") as fh: fh.write(f"{pd.Timestamp.now():%H:%M:%S} {msg}\n")
    print(msg, flush=True)

def frame(tks=ALL):
    cp = f"{HERE}/cache_frame.parquet"
    if os.path.exists(cp):
        D = pd.read_parquet(cp); 
        if set(tks) <= set(D.tk.unique()): return D[D.tk.isin(tks)].reset_index(drop=True)
    rows = []
    for tk in tks:
        d = rlib.ticker_frame(tk).copy()
        y = liq.overnight_targets(tk)
        d["y_oo"] = y["y_oo"].reindex(d.index).values
        d.index.name = "date"; rows.append(d.reset_index())
    D = pd.concat(rows); D = D[(D.date >= "2016-06-01") & (D.date <= DEV_END)]
    D = rlib.add_market(D)
    # minute/official prices are unadjusted for dividends: add the ex-date dividend (panel 'div', known in advance)
    z = np.load(f"{ROOT}/data/panel_liquid.npz", allow_pickle=True); pdx = pd.DatetimeIndex(z["dates"]); ti = {t: i for i, t in enumerate(z["tickers"])}
    parts = []
    for tk in tks:
        dv = pd.Series(z["div"][:, ti[tk]].astype(float), pdx); cl = pd.Series(z["close"][:, ti[tk]].astype(float), pdx)
        parts.append(pd.DataFrame({"date": pdx, "tk": tk, "dvy": (dv.shift(-1) / cl).fillna(0.0).values}))
    D = D.merge(pd.concat(parts), on=["date", "tk"], how="left"); D["dvy"] = D.dvy.fillna(0.0)
    for c in ["y_oo", "y_on", "y_cc"]: D[c] = D[c] + D.dvy
    D.loc[D.date >= DEV_END, ["y_oo", "y_on", "y_cc", "y_day", "y_on60"]] = np.nan     # exit day after DEV_END
    D.reset_index(drop=True).to_parquet(cp); return D.reset_index(drop=True)

def vix_ratio_lag(idx):
    F = pd.read_csv(f"{ROOT}/agents/vol/fred_vix.csv", index_col=0, parse_dates=True).loc[:DEV_END]
    r = (F.VIX / F.VIX3M).ffill()
    return r.shift(1).reindex(pd.DatetimeIndex(idx), method="ffill")   # FRED close = 16:15 -> use t-1

def run_book(W, R, cost_bps=1.5, borrow=0.01, comm=True):
    """W, R: date x tk weights decided at close t and overnight returns t->t+1 (indexed at decision date t).
    net = sum w r - 2*|w|*cost - short*borrow/252 - 2*npos*COMM_BPS.  Indexed at decision date (rlib.book convention)."""
    W = W.fillna(0.0); Rf = R.reindex_like(W)
    W = W.where(Rf.notna(), 0.0)
    g = (W * Rf.fillna(0)).sum(1)
    c = W.abs().sum(1) * 2 * cost_bps / 1e4 + W.clip(upper=0).abs().sum(1) * borrow / 252
    npos = (W != 0).sum(1)
    cm = 2 * npos * COMM_BPS / 1e4 if comm else 0.0
    n = (g - c - cm).loc[START:DEV_END]
    return n, g.loc[START:DEV_END], npos.loc[START:DEV_END], W.abs().sum(1).loc[START:DEV_END]

_REF = None
EXIT_DATED = ("vixon_ls", "mom_E4", "SPYcc")   # these ref series are booked on the exit day; ours (like rlib.book, onml3, xsr3) on the decision day
def refs():
    global _REF
    if _REF is None:
        fs = {"vixon_ls": "agents/etfmacro/cands/vixon_r95_ls.csv", "onml3_k6": "agents/onml3/cands/onml3_f26calx_q3_lam3_moo_k6.csv",
              "xsr3_lo10": "agents/xsrule3/cands/xsr3_cAx_u50_moo_lo10.csv", "mom_E4": "logs/mom_grid/mom_E4_top25.csv"}
        _REF = {k: pd.read_csv(f"{ROOT}/{v}", index_col=0, parse_dates=True).iloc[:, 0].loc[:DEV_END] for k, v in fs.items()}
        sp = rlib.ticker_frame("SPY")["close"].astype(float); sp.index = pd.DatetimeIndex(sp.index)
        _REF["SPYcc"] = sp.pct_change().loc[START:DEV_END]
    return _REF

def corrs(r):
    out = {}
    for k, s in refs().items():
        rr = r.shift(1) if k in EXIT_DATED else r          # align the same physical night/day
        a = pd.concat([rr, s], axis=1).dropna(); a = a[(a.iloc[:, 0] != 0)]
        out[k] = round(float(a.iloc[:, 0].corr(a.iloc[:, 1])), 2) if len(a) > 50 else np.nan
    return out

def report(name, n, npos=None, gross=None, fold=True):
    st = stats(n, name, p=False)
    fo = fold_oos({name: n}, p=False)[0] if fold else None
    line = (f"{name:38s} Sh {st['sharpe']:5.2f} foldOOS {sh(fo):5.2f} | ret {st['ret']*100:5.1f}% vol {st['vol']*100:4.1f}% mdd {st['mdd']*100:6.1f}% "
            f"| h {st['h1']:.2f}/{st['h2']:.2f}")
    if npos is not None: line += f" | pos/day {npos.mean():.2f} gross {gross.mean():.2f}"
    yr = n.groupby(n.index.year).apply(sh); line += " | yrs " + " ".join(f"{v:+.1f}" for v in yr.values)
    line += " | corr " + " ".join(f"{k}:{v:+.2f}" for k, v in corrs(n).items())
    log(line); return st, fo
