"""Exp 7: sensitivity of the final config: a (z30 weight), cap, dz, z-window, decision bin."""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from final import *
A = ["SPY", "QQQ", "NVDA", "MSFT", "AMZN", "AAPL", "TSLA"]
base = dict(a=0.5, cap=1.5, dz=1.0, dec=389, zw=20)
grid = [("a", [0.0, 0.25, 0.5, 0.75, 1.0]), ("cap", [1.0, 1.5, 2.0, 3.0]), ("dz", [0.5, 1.0, 1.5]), ("zw", [10, 20, 40, 60]), ("dec", [385, 388, 389])]
Ms = {t: Min(t) for t in A}
for k, vals in grid:
    for v in vals:
        kw = {**base, k: v}; ds = {}
        for t in A:
            M = Ms[t]; ms = M.insample; g, n, tr = pathbt(M, build(M, **kw))
            ds[t] = stats(M.dates, g, n, None, t, ms, p=False)
        combo(ds, f"{k}={v}")
    print()
# single-ticker robustness at base config: per-ticker ON-only for the 1.5bps ETFs (overnight leg only)
for t in ["IWM", "DIA", "XLK", "XLF", "XLE", "GLD", "TLT"]:
    d = run_ticker(t, **{**base, "legs": ("ON",)}); line(t + " ON-only", d)
