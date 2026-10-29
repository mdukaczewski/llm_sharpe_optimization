"""VIXY overnight carry conditioned on SPY/VIXY intraday signals (s = 0.5*z30 - z60 of day t, known at close t); VRP family."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
D = load_etf(); idx = D["ret"].index; F = load_fred(idx); ratio = (F.VIX / F.VIX3M); vix = F.VIX / 100
rv10 = vlib.rv(D["ret"]["SPY"], 10); vrp = vix - rv10
d = tf("VIXY").reindex(idx); s = tf("SPY").reindex(idx)
wd = pd.Series(idx.weekday, index=idx); notmon = (wd != 4).astype(float); f95 = (ratio < 0.95).astype(float)
def leg(fr, w, y="y_on", cost=1.5, borrow=0.02, shift=1):
    w = w.reindex(idx).fillna(0.0); r = fr[y]
    g = w * r; c = w.abs() * 2 * cost / 1e4 + w.clip(upper=0).abs() * borrow / 252
    n = (g - c).shift(shift).fillna(0.0).loc[START:DEV_END]; n[fr[y].shift(shift).isna()] = 0.0; return n
for nm in ["spy_ovn", "qqq_ovn", "spy_tom", "mr_spy"]:
    q = pd.read_csv(f"sleeves/{nm}.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[:DEV_END]; stats(q, f"existing sleeve {nm}")
ss = (0.5 * s.z30 - s.z60).clip(-5, 5).fillna(0); sv = (0.5 * d.z30 - d.z60).clip(-5, 5).fillna(0)
print("== VIXY overnight (short) by SPY signal s bucket, all days / contango days")
on_next = d.y_on
for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 9)]:
    m = (ss >= lo) & (ss < hi); m2 = m & (f95 > 0) & (notmon > 0)
    print(f"  s_spy in [{lo},{hi}) n={m.sum()} VIXY on {on_next[m].mean()*1e4:6.1f}bp Sh {sh(on_next[m]):5.2f} | contango&notFri n={m2.sum()} {on_next[m2].mean()*1e4:6.1f}bp Sh {sh(on_next[m2]):5.2f} | SPY on {s.y_on[m].mean()*1e4:5.1f}bp")
for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 9)]:
    m = (sv >= lo) & (sv < hi); m2 = m & (f95 > 0) & (notmon > 0)
    print(f"  s_vixy in [{lo},{hi}) n={m.sum()} VIXY on {on_next[m].mean()*1e4:6.1f}bp Sh {sh(on_next[m]):5.2f} | contango&notFri n={m2.sum()} {on_next[m2].mean()*1e4:6.1f}bp Sh {sh(on_next[m2]):5.2f}")
for col in ["z60", "z30", "zoc", "zcc", "zgap"]:
    for src, fr in [("spy", s), ("vixy", d)]:
        z = fr[col].clip(-5, 5).fillna(0); m2 = (f95 > 0) & (notmon > 0)
        print(f"  {src}.{col:5s} corr with next VIXY overnight: all {z.corr(on_next):+.3f}  contango {z[m2].corr(on_next[m2]):+.3f}")
res = {}
res["r95"] = leg(d, -0.5 * notmon * f95); stats(res["r95"], "r95 base")
res["vrp0"] = leg(d, -0.5 * notmon * (vrp > 0)); stats(res["vrp0"], "VRP>0 skip Fri")
res["r95_vrp0"] = leg(d, -0.5 * notmon * f95 * (vrp > 0)); stats(res["r95_vrp0"], "r95 & VRP>0")
res["r95_or_vrp"] = leg(d, -0.5 * notmon * ((f95 > 0) | (vrp > 0.03))); stats(res["r95_or_vrp"], "r95 | VRP>3%")
res["r100"] = leg(d, -0.5 * notmon * (ratio < 1.0)); stats(res["r100"], "ratio<1.0 skip Fri")
print("== add SPY signal tilt: w = -0.5 * f95 * notmon * clip(1 + a*s_spy, 0, 2)")
for a in [0.25, 0.5]:
    res[f"r95_tilt{a}"] = leg(d, -0.5 * notmon * f95 * (1 + a * ss).clip(0, 2)); stats(res[f"r95_tilt{a}"], f"r95 tilt a={a}")
    stats(leg(d, -0.5 * notmon * f95 * (ss > -1)), f"r95 only if s_spy > -1")
    stats(leg(d, -0.5 * notmon * f95 * (1 + a * sv).clip(0, 2)), f"r95 tilt by VIXY own s, a={a}")
res["r95_sgt0"] = leg(d, -0.5 * notmon * f95 * (ss > 0)); stats(res["r95_sgt0"], "r95 only if s_spy > 0")
stats(leg(d, -0.5 * notmon * f95 * (ss <= 0)), "r95 only if s_spy <= 0")
print("== fold-OOS")
oos, tab = fold_oos(res, label="family x8"); print(tab.to_string())
for k, v in res.items(): fold_oos(v, label=k)
pd.DataFrame(res).to_csv("/home/user/llm_sharpe_optimization/agents/etfmacro/x8_res.csv")
