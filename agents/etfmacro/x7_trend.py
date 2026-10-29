"""Cross-asset trend / carry, weekly rebalance, daily panel, vlib.backtest (decide close t, fill close t+1), 1.5 bps."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
P = panel(); TK = ["TLT", "IEF", "GLD", "SLV", "HYG", "LQD", "EEM", "EFA", "USO", "GDX", "SPY", "QQQ", "IWM", "XLE", "XLU", "XLF", "XLK", "DBC", "UUP", "FXE"]
TK = [t for t in TK if t in P.ix]; print("tickers", TK)
cl = pd.DataFrame({t: pseries(P, t) for t in TK}); dv = pd.DataFrame({t: pseries(P, t, "div") for t in TK}).fillna(0)
ret = (cl + dv) / cl.shift(1) - 1; ret = ret.loc["2015-01-01":]
D = {"ret": ret, "close": cl}
vol = ret.rolling(60).std() * np.sqrt(252)
wk = pd.Series(ret.index.weekday, index=ret.index)
res = {}
def run(name, W, cost=1.5):
    W = W.where(wk == 4, np.nan).ffill().fillna(0.0)          # decide Friday close, fill Monday close, hold a week
    b = bt(W, D, cost_bps=cost); n = b["net"]; stats(n, name); res[name] = n; return n
for lb in [63, 126, 252]:
    mom = cl / cl.shift(lb) - 1
    sig = np.sign(mom)
    W = (sig / vol).div((1 / vol).sum(1), axis=0) * 0.5    # inverse-vol, gross 0.5... normalise sum|w| = 1
    W = W.div(W.abs().sum(1), axis=0)
    run(f"tsmom{lb} all L/S invvol", W)
    run(f"tsmom{lb} long-only", W.clip(lower=0))
    # cross-sectional: long top 5 short bottom 5 by vol-adjusted momentum
    sc = (mom / vol).rank(axis=1); n = sc.notna().sum(1)
    L = (sc > n.values[:, None] - 5).astype(float); S = (sc <= 5).astype(float)
    run(f"xsmom{lb} top5/bot5", L / 5 - S / 5)
# combined 3/6/12 trend
mom = sum(np.sign(cl / cl.shift(l) - 1) for l in [63, 126, 252]) / 3
W = (mom / vol); W = W.div(W.abs().sum(1), axis=0); run("tsmom 3/6/12 blend", W)
W2 = (mom.clip(lower=0) / vol); W2 = W2.div(W2.abs().sum(1), axis=0).fillna(0); run("tsmom blend long-only", W2)
# defensive: hold TLT/GLD/IEF when their 10m mom > 0
sub = ["TLT", "IEF", "GLD"]; m2 = (cl[sub] / cl[sub].shift(210) - 1 > 0).astype(float) / 3; run("TLT/IEF/GLD 10m mom long-only", m2.reindex(columns=TK).fillna(0))
print("== fold-OOS"); fold_oos(res, label="trend family")
for k, v in res.items(): fold_oos(v, label=k)
