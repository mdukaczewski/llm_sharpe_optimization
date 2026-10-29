"""Exp 6: overnight-leg sizing variants (single formula for all tickers) + conservative fill,
then the chain with the best variant; correlation structure of ticker books."""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from ilib import *
import warnings; warnings.filterwarnings("ignore")
LAG = lambda x: np.r_[np.nan, x[:-1]]
def zsd(x, w=20):
    s = pd.Series(x); return (s / s.rolling(w, min_periods=10).std().shift(1)).values

def build(M, legs, l1w=None, dz=1.0, dec=389, zspy=None):
    """l1w(z60, z30, zspy) -> weight array. dec: decision bin for the overnight entry (389 -> MOC at
    official close; 388 -> market order filled at the 15:59 bar close)."""
    nd = len(M.dates); pos = np.zeros((nd, 391))
    if "L1" in legs:
        z60 = np.nan_to_num(zsd(M.r(dec - 60, dec))); z30 = np.nan_to_num(zsd(M.r(-1, 29)))
        w = np.nan_to_num(l1w(z60, z30, zspy))
        if dec == 389: pos[:, 390] += w
        else: pos[:, 389] += w; pos[:, 390] += w
        pos[1:, 0] += w[:-1]
    if "L2" in legs:
        z = np.nan_to_num(zsd(M.r(-1, 329))); s = np.sign(z) * (np.abs(z) > dz); pos[:, 330:360] += s[:, None]
    if "L4" in legs:
        z = np.nan_to_num(zsd(LAG(M.r(-1, 29)))); s = -np.sign(z) * (np.abs(z) > dz); pos[:, 1:30] += s[:, None]
    return pos

def rep(M, name, pos, store):
    ms = M.insample & ~M.half
    g, n, t = pathbt(M, pos)
    d = stats(M.dates, g, n, None, f"{M.tk} {name}", ms, p=False); d["trades"] = t[ms].sum() / (ms.sum() / 252)
    print(f"{M.tk+' '+name:34s} Sh gross {d['sh_g']:5.2f} net {d['sh_n']:5.2f} | ret {d['ret']*100:6.2f}% vol {d['vol']*100:5.2f}% mdd {d['mdd']*100:6.1f}% | trades/yr {d['trades']:5.0f} | yrs " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in d["yr"].items()), flush=True)
    store[name] = d; return d

def combo_report(legs, tks, name):
    tks = [t for t in tks if t in legs]
    port = combine([legs[t]["n"] for t in tks]); pg = combine([legs[t]["g"] for t in tks])
    pstats(port, f"COMBO[{name}]", pg)
    h1 = port[port.index < "2021-10-28"]; h2 = port[port.index >= "2021-10-28"]
    print(f"      halves: Sh net 2016-21 {sh(h1):.2f}  2021-26 {sh(h2):.2f}")
    return port

VARS = {
    "base cap1.5":  lambda z60, z30, zs: np.clip(-z60, 0, 1.5),
    "floor0.5":     lambda z60, z30, zs: np.clip(0.5 - z60, 0, 1.5),
    "floor0.5 cap1":lambda z60, z30, zs: np.clip(0.5 - z60, 0, 1.0),
    "+0.5*z30":     lambda z60, z30, zs: np.clip(0.5 * z30 - z60, 0, 1.5),
    "own+spy":      lambda z60, z30, zs: np.clip(-(z60 + zs) / 2, 0, 1.5),
    "floor+spy":    lambda z60, z30, zs: np.clip(0.5 - (z60 + zs) / 2, 0, 1.5),
}
if __name__ == "__main__":
    A = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SPY", "QQQ", "NVDA", "MSFT", "AMZN", "AAPL", "TSLA"]
    S = Min("SPY"); zspy = np.nan_to_num(zsd(S.r(329, 389))); zspy = pd.Series(zspy, index=S.dates)
    res = {}
    for tk in A:
        M = Min(tk); R = {}; res[tk] = R
        zs = zspy.reindex(M.dates).fillna(0).values
        for k, f in VARS.items(): rep(M, f"L1 {k}", build(M, ("L1",), f, zspy=zs), R)
        rep(M, "L1 base dec388", build(M, ("L1",), VARS["base cap1.5"], dec=388, zspy=zs), R)
        rep(M, "L1 floor0.5 dec388", build(M, ("L1",), VARS["floor0.5"], dec=388, zspy=zs), R)
        for k in ["base cap1.5", "floor0.5", "+0.5*z30", "floor+spy"]:
            rep(M, f"chain {k}", build(M, ("L1", "L2", "L4"), VARS[k], zspy=zs), R)
        rep(M, "chain floor0.5 dec388", build(M, ("L1", "L2", "L4"), VARS["floor0.5"], dec=388, zspy=zs), R)
        print()
    print("=== COMBOS")
    for name in res[A[0]]:
        combo_report({t: res[t][name] for t in A}, A, f"A {name}")
    print("\n=== correlation of ticker books (chain floor0.5, net)")
    df = pd.concat({t: res[t]["chain floor0.5"]["n"] for t in A}, axis=1)
    pd.set_option("display.width", 200); print(df.corr().round(2))
