"""Exp 5: cheap-universe portfolio of netted position paths.
 L1  overnight long, sized w = clip(-z60, 0, cap) (z60 = last-60-min return / trailing 20d sd), exit 09:31
 L1s overnight long if last60<0 (unsized), for comparison
 L2  15:00-15:30 momentum: sign(open->15:00) if |z| > dz
 L4  09:31-10:00 reversal of yesterday's first-30: -sign if |z| > dz
Universe A (cheap per unit vol): SPY QQQ NVDA MSFT AMZN AAPL TSLA ; B = A + IWM XLK (overnight only)."""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from ilib import *
import warnings; warnings.filterwarnings("ignore")
LAG = lambda x: np.r_[np.nan, x[:-1]]
def zsd(x, w=20):
    s = pd.Series(x); return (s / s.rolling(w, min_periods=10).std().shift(1)).values

def build(M, legs, cap=1.5, dz=0.5, w1=60, sized=True):
    nd = len(M.dates); pos = np.zeros((nd, 391))
    if "L1" in legs:
        last = M.r(389 - w1, 389)
        side = np.clip(-np.nan_to_num(zsd(last)), 0, cap) if sized else (last < 0) * 1.0
        pos[:, 390] += side; pos[1:, 0] += side[:-1]
    if "L2" in legs:
        z = np.nan_to_num(zsd(M.r(-1, 329))); s = np.sign(z) * (np.abs(z) > dz)
        pos[:, 330:360] += s[:, None]
    if "L4" in legs:
        z = np.nan_to_num(zsd(LAG(M.r(-1, 29)))); s = -np.sign(z) * (np.abs(z) > dz)
        pos[:, 1:30] += s[:, None]
    return pos

def rep(M, name, pos, store):
    ms = M.insample & ~M.half
    g, n, t = pathbt(M, pos)
    d = stats(M.dates, g, n, None, f"{M.tk} {name}", ms, p=False); d["trades"] = t[ms].sum() / (ms.sum() / 252)
    print(f"{M.tk+' '+name:34s} Sh gross {d['sh_g']:5.2f} net {d['sh_n']:5.2f} | ret {d['ret']*100:6.2f}% vol {d['vol']*100:5.2f}% mdd {d['mdd']*100:6.1f}% | trades/yr {d['trades']:5.0f} | yrs " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in d["yr"].items()), flush=True)
    store[name] = d; return d

def combo_report(legs, tks, name):
    rets = [legs[t]["n"] for t in tks if t in legs]; gs = [legs[t]["g"] for t in tks if t in legs]
    port = combine(rets); pg = combine(gs)
    pstats(port, f"COMBO[{name}]", pg)
    h1 = port[port.index < "2021-10-28"]; h2 = port[port.index >= "2021-10-28"]
    print(f"      halves: Sh net 2016-21 {sh(h1):.2f}  2021-26 {sh(h2):.2f}")
    return port

if __name__ == "__main__":
    A = ["SPY", "QQQ", "NVDA", "MSFT", "AMZN", "AAPL", "TSLA"]; ON_ONLY = ["IWM", "XLK", "XLF", "DIA", "GLD"]
    res = {}
    for tk in A + ON_ONLY:
        M = Min(tk); S = {}; res[tk] = S
        rep(M, "L1s", build(M, ("L1",), sized=False), S)
        for cap in [1.0, 1.5, 2.0]: rep(M, f"L1 cap={cap}", build(M, ("L1",), cap=cap), S)
        if tk in A:
            for dz in [0.0, 0.5, 1.0]:
                rep(M, f"L2 dz={dz}", build(M, ("L2",), dz=dz), S)
                rep(M, f"L4 dz={dz}", build(M, ("L4",), dz=dz), S)
            rep(M, "L1+L2 dz=0.5", build(M, ("L1", "L2")), S)
            rep(M, "L1+L4 dz=0.5", build(M, ("L1", "L4")), S)
            rep(M, "L1+L2+L4 dz=0.5", build(M, ("L1", "L2", "L4")), S)
            rep(M, "L1+L2+L4 dz=1.0", build(M, ("L1", "L2", "L4"), dz=1.0), S)
        print()
    print("=== COMBOS (equal-vol, trailing 60d)")
    for name in ["L1s", "L1 cap=1.5"]:
        combo_report({t: res[t][name] for t in res}, A, f"A {name}")
        combo_report({t: res[t][name] for t in res}, A + ON_ONLY, f"A+ON {name}")
        combo_report({t: res[t][name] for t in res}, ["SPY", "QQQ"], f"SPY+QQQ {name}")
    for name in ["L2 dz=0.5", "L4 dz=0.5", "L1+L2 dz=0.5", "L1+L4 dz=0.5", "L1+L2+L4 dz=0.5", "L1+L2+L4 dz=1.0"]:
        combo_report({t: res[t][name] for t in A}, A, f"A {name}")
    # mixed book: full chain in A, overnight-only (sized) in ON_ONLY
    mixed = {t: res[t]["L1+L2+L4 dz=0.5"] for t in A}; mixed.update({t: res[t]["L1 cap=1.5"] for t in ON_ONLY})
    port = combo_report(mixed, A + ON_ONLY, "MIXED chain(A)+ON(others)")
    port.to_csv("/home/trumpet/strat20/agents/intraday/logs/exp5_mixed_daily.csv")
