"""Fix: prior-day cc known at close t-1 is y_cc.shift(2). Redo prev-day conditioning. Plus sector-ETF XS overnight test."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
for tk in ["VIXY", "GLD", "TLT", "SPY", "IEF", "HYG", "EEM", "SVXY"]:
    e = tf(tk); e = e[~e.half]; o = e.y_on.shift(1); dy = e.y_day; pcc = e.y_cc.shift(2); sd = e.y_cc.rolling(20).std().shift(2)
    z = pcc / sd
    pon = e.y_on.shift(2); pday = e.y_day.shift(1)   # prior overnight (close t-2 -> 09:31 t-1) and prior intraday (t-1), both known at close t-1
    print(f"{tk}: overnight all {o.mean()*1e4:5.1f}bp Sh {sh(o):5.2f}")
    for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 9)]:
        m = (z >= lo) & (z < hi); print(f"   prevcc z[{lo},{hi}) n={m.sum()} on {o[m].mean()*1e4:6.1f}bp Sh {sh(o[m]):5.2f} | day {dy[m].mean()*1e4:6.1f} Sh {sh(dy[m]):5.2f}")
    zd = pday / e.y_day.rolling(20).std().shift(1)
    for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 9)]:
        m = (zd >= lo) & (zd < hi); print(f"   prev intraday z[{lo},{hi}) n={m.sum()} on {o[m].mean()*1e4:6.1f}bp Sh {sh(o[m]):5.2f} | day {dy[m].mean()*1e4:6.1f} Sh {sh(dy[m]):5.2f}")
    zo = pon / e.y_on.rolling(20).std().shift(2)
    for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 9)]:
        m = (zo >= lo) & (zo < hi); print(f"   prev overnight z[{lo},{hi}) n={m.sum()} on {o[m].mean()*1e4:6.1f}bp Sh {sh(o[m]):5.2f} | day {dy[m].mean()*1e4:6.1f} Sh {sh(dy[m]):5.2f}")
# ---- sector ETF cross-sectional overnight reversal
ETFS = ["XLF", "XLK", "XLE", "XLU", "XLV", "XLI", "XLY", "XLP", "SMH", "KRE", "GDX", "EEM", "EFA", "IWM", "DIA", "QQQ", "SPY", "HYG", "TLT", "GLD", "USO", "SLV", "IEF"]
rows = []
for tk in ETFS:
    e = tf(tk); e = e[~e.half]; e["tk"] = tk; rows.append(e.reset_index())
Dd = pd.concat(rows); Dd = Dd[Dd.date <= DEV_END]
Dd["s"] = 0.5 * Dd.z30 - Dd.z60
Dd = rlib.clean(Dd, ["s", "z30", "z60", "z120", "zoc", "zcc", "cc5"])
for sub, lbl in [(ETFS[:11], "11 sectors+"), (ETFS[:13], "13 eq ETFs"), (ETFS, "23 ETFs")]:
    S = Dd[Dd.tk.isin(sub)]
    for k in [2, 3, 4]:
        for y in ["y_on", "y_cc"]:
            for sig in ["s", "z60", "zcc", "z30"]:
                p = S[sig] if sig != "z60" else -S["z60"]
                if sig == "zcc": p = -S["zcc"]
                n = rlib.book(S, p, k=k, y=y, mode="ls", cost=1.5, p=False)
                nl = rlib.book(S, p, k=k, y=y, mode="lo", cost=1.5, p=False)
                print(f"{lbl:12s} k={k} {y:5s} sig={sig:4s} LS Sh {sh(n):5.2f} h1/h2 {sh(n[n.index<'2021-10-28']):5.2f}/{sh(n[n.index>='2021-10-28']):5.2f} | LO Sh {sh(nl):5.2f}")
