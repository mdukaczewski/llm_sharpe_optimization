"""More facts: VIXY overnight by ratio x half, weekday x half, rv-iv, tails; GLD/TLT overnight conditioning; sector ETF XS overnight."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
D = load_etf(); idx = D["ret"].index; F = load_fred(idx); ratio = F.VIX / F.VIX3M
rv10 = vlib.rv(D["ret"]["SPY"], 10); rv20 = vlib.rv(D["ret"]["SPY"], 20)
d = tf("VIXY"); d = d[~d.half]
on_t = d.y_on.shift(1)   # overnight ending at 09:31 of day t (decision at close t-1)
day = d.y_day            # 09:32 -> close of day t (decision 09:31 of t)
h = d.index < "2021-10-28"
rt = ratio.reindex(d.index).shift(1); rvs = (F.VIX.reindex(d.index) / 100 - rv10.reindex(d.index)).shift(1)   # VIX - realized, known at close t-1
wd = pd.Series(d.index.weekday, index=d.index)
def row(m, lbl):
    print(f"  {lbl:28s} n={int(m.sum()):4d} | on {on_t[m].mean()*1e4:6.1f}bp Sh {sh(on_t[m]):5.2f} [h1 {sh(on_t[m & h]):5.2f} h2 {sh(on_t[m & ~h]):5.2f}] | day {day[m].mean()*1e4:6.1f}bp Sh {sh(day[m]):5.2f} [h1 {sh(day[m & h]):5.2f} h2 {sh(day[m & ~h]):5.2f}]")
print("VIXY by ratio bucket (lagged):")
for lo, hi in [(0, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0), (1.0, 1.05), (1.05, 9)]: row((rt >= lo) & (rt < hi), f"ratio[{lo},{hi})")
print("VIXY by VIX-RV10 spread (lagged):")
for lo, hi in [(-9, -0.05), (-0.05, 0), (0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 9)]: row((rvs >= lo) & (rvs < hi), f"vix-rv10[{lo},{hi})")
print("VIXY by weekday:")
for w in range(5): row(wd == w, f"weekday {w}")
print("VIXY by weekday, ratio<0.95:")
for w in range(5): row((wd == w) & (rt < 0.95), f"weekday {w} ratio<.95")
print("VIXY by VIX level (lagged):")
vx = F.VIX.reindex(d.index).shift(1)
for lo, hi in [(0, 13), (13, 16), (16, 20), (20, 25), (25, 35), (35, 99)]: row((vx >= lo) & (vx < hi), f"VIX[{lo},{hi})")
print("VIXY by prior-day VIXY cc (lagged):")
pc = d.y_cc.shift(1)
for lo, hi in [(-9, -0.05), (-0.05, -0.02), (-0.02, 0), (0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 9)]: row((pc >= lo) & (pc < hi), f"prevcc[{lo},{hi})")
print("tails of VIXY overnight: worst 10"); print((on_t.sort_values(ascending=False).head(10) * 100).round(1).to_string())
print("VIXY overnight autocorr with prev overnight / prev intraday:", on_t.corr(on_t.shift(1)).round(3), on_t.corr(day.shift(1)).round(3))
# ---------- GLD/TLT/SPY overnight conditioning
for tk in ["GLD", "TLT", "SPY", "IEF", "HYG", "EEM"]:
    e = tf(tk); e = e[~e.half]; o = e.y_on.shift(1); dy = e.y_day; pcc = e.y_cc.shift(1); w2 = pd.Series(e.index.weekday, index=e.index)
    sd = e.y_cc.rolling(20).std().shift(1)
    print(f"{tk}: overnight all {o.mean()*1e4:5.1f}bp Sh {sh(o):5.2f} h1 {sh(o[e.index<'2021-10-28']):5.2f} h2 {sh(o[e.index>='2021-10-28']):5.2f} | intraday Sh {sh(dy):5.2f}")
    for w in range(5): m = w2 == w; print(f"   wd{w} on {o[m].mean()*1e4:5.1f}bp Sh {sh(o[m]):5.2f} | day {dy[m].mean()*1e4:5.1f} Sh {sh(dy[m]):5.2f}")
    for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 9)]:
        m = ((pcc / sd) >= lo) & ((pcc / sd) < hi); print(f"   prevcc z[{lo},{hi}) n={m.sum()} on {o[m].mean()*1e4:5.1f}bp Sh {sh(o[m]):5.2f} | day {dy[m].mean()*1e4:5.1f} Sh {sh(dy[m]):5.2f}")
