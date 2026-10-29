"""Fact-finding: VIX ETP overnight vs intraday return split, and conditioning."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
D = load_etf(); F = load_fred(D["ret"].index); ratio = F.VIX / F.VIX3M
for tk in ["VIXY", "VXX", "VIXM", "SVXY", "UVXY", "SPY", "TLT", "GLD"]:
    d = tf(tk); d = d[~d.half]
    on = d.y_on; day = d.y_day; cc = d.y_cc
    # y_day of day t+1 is aligned at t+1 index; y_on at t is close t->09:31 t+1. Use raw per-day: overnight ending day t = y_on.shift(1)
    on_t = on.shift(1); 
    print(f"{tk:5s} n={len(d)} | overnight(close->09:31) mean {on_t.mean()*1e4:6.1f}bp Sh {sh(on_t):5.2f} | intraday(09:32->close) mean {day.mean()*1e4:6.1f}bp Sh {sh(day):5.2f} | cc mean {cc.mean()*1e4:6.1f}bp Sh {sh(cc):5.2f}")
    if tk in ["VIXY", "VXX", "SVXY"]:
        rt = ratio.reindex(d.index)
        for lo, hi in [(0, 0.85), (0.85, 0.95), (0.95, 1.05), (1.05, 9)]:
            m = (rt.shift(1) >= lo) & (rt.shift(1) < hi)   # ratio known at close t-1 (both legs on day t need decision at close t-1 / 09:31 t)
            print(f"   ratio[{lo},{hi}) n={m.sum():4d} | on {on_t[m].mean()*1e4:6.1f}bp Sh {sh(on_t[m]):5.2f} | day {day[m].mean()*1e4:6.1f}bp Sh {sh(day[m]):5.2f}")
        wd = pd.Series(d.index.weekday, index=d.index)
        for w in range(5):
            m = wd == w
            print(f"   weekday {w} | on {on_t[m].mean()*1e4:6.1f}bp Sh {sh(on_t[m]):5.2f} | day {day[m].mean()*1e4:6.1f}bp Sh {sh(day[m]):5.2f}")
        for a, b in [("2016", "2019"), ("2020", "2021"), ("2022", "2023"), ("2024", "2026")]:
            m = (d.index >= a) & (d.index < str(int(b) + 1))
            print(f"   {a}-{b} | on {on_t[m].mean()*1e4:6.1f}bp Sh {sh(on_t[m]):5.2f} | day {day[m].mean()*1e4:6.1f}bp Sh {sh(day[m]):5.2f}")
