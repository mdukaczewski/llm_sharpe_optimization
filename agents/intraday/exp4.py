"""Exp 4: full-day position paths per ticker (netted costs) built from the robust effects:
  L1 overnight: long at close if last-w-min return < 0 (exit at 09:31 fill = c[1])
  L2 15:00-15:30: sign(open->15:00) ; L3 15:30-close: sign(14:30->15:30)
  L4 09:31-10:00: -sign(prev day first30)
Variants of L1 window / threshold / vol-scaling; then cross-ticker combos."""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from ilib import *
import warnings; warnings.filterwarnings("ignore")
LAG = lambda x: np.r_[np.nan, x[:-1]]

def build(M, L1w=60, L1thr=0.0, use=("L1",), L1size=None, exit_bin=0, l4_hold_on=False):
    nd = len(M.dates); pos = np.zeros((nd, 391))
    if "L1" in use:
        last = M.r(389 - L1w, 389)
        side = (last < -L1thr * np.nan_to_num(pd.Series(last).rolling(20).std().shift(1).values)) * 1.0
        if L1size is not None: side = side * L1size
        pos[:, 390] += side
        for k in range(0, exit_bin + 1):              # exit decided at bin exit_bin -> filled c[exit_bin+1]
            pos[1:, k] += side[:-1]
    if "L2" in use:
        s = np.sign(np.nan_to_num(M.r(-1, 329)))
        pos[:, 330:360] += s[:, None]                  # decided 329, hold 330..359 (fills c330 -> c360)
    if "L3" in use:
        s = np.sign(np.nan_to_num(M.r(329, 359)))
        pos[:, 360:390] += s[:, None]                  # decided 359, fill c360, exit fill c390
    if "L4" in use:
        s = -np.sign(np.nan_to_num(LAG(M.r(-1, 29))))
        pos[:, 1:30] += s[:, None]                     # decided bin 0, fill c1, exit decided 29 fill c30
    return pos

def rep(M, name, pos, store=None):
    ms = M.insample & ~M.half
    g, n, t = pathbt(M, pos)
    d = stats(M.dates, g, n, None, f"{M.tk} {name}", ms); d["trades"] = t[ms].sum() / (ms.sum() / 252)
    if store is not None: store[name] = d
    return d

if __name__ == "__main__":
    tks = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SPY","QQQ","IWM","DIA","XLF","XLK","XLE","TLT","GLD","AAPL","MSFT","NVDA","AMZN","TSLA"]
    mode = sys.argv[2] if len(sys.argv) > 2 else "A"
    legs = {}
    for tk in tks:
        M = Min(tk); S = {}; legs[tk] = S
        print(f"--- {tk}")
        if mode == "A":
            for w in [30, 60, 90, 120]: rep(M, f"L1 w={w}", build(M, L1w=w), S)
            for thr in [0.25, 0.5, 1.0]: rep(M, f"L1 w=60 thr={thr}sd", build(M, L1thr=thr), S)
            rep(M, "L1 w=60 exit10:00", build(M, exit_bin=29), S)
            v = pd.Series(M.cc).rolling(20).std().shift(1).values * np.sqrt(252)
            rep(M, "L1 w=60 volscaled", build(M, L1size=np.clip(0.15 / np.nan_to_num(v, nan=0.15), 0, 2)), S)
            rep(M, "L1 w=60 lowvol-only", build(M, L1size=(v < np.nanmedian(v)) * 1.0), S)
            rep(M, "L1 w=60 highvol-only", build(M, L1size=(v >= np.nanmedian(v)) * 1.0), S)
        else:
            rep(M, "L1", build(M, use=("L1",)), S)
            rep(M, "L2", build(M, use=("L2",)), S)
            rep(M, "L3", build(M, use=("L3",)), S)
            rep(M, "L4", build(M, use=("L4",)), S)
            rep(M, "L2+L3", build(M, use=("L2", "L3")), S)
            rep(M, "L1+L3", build(M, use=("L1", "L3")), S)
            rep(M, "L1+L2+L3", build(M, use=("L1", "L2", "L3")), S)
            rep(M, "L1+L2+L3+L4", build(M, use=("L1", "L2", "L3", "L4")), S)
            rep(M, "L1+L4", build(M, use=("L1", "L4")), S)
    print("\n=== cross-ticker equal-vol combos (trailing 60d vol, 10% target)")
    for name in legs[tks[0]]:
        rets = [legs[t][name]["n"] for t in tks]; gs = [legs[t][name]["g"] for t in tks]
        pstats(combine(rets), f"COMBO[{len(tks)}] {name}", combine(gs))
        e = [t for t in tks if t in ETF4]
        if len(e) == 4: pstats(combine([legs[t][name]["n"] for t in e]), f"COMBO[ETF4] {name}", combine([legs[t][name]["g"] for t in e]))
