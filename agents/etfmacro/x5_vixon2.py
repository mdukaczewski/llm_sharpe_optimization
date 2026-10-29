"""VIXY overnight carry: vol-scaled sizing; intraday leg in flat-ratio regimes; combination."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
D = load_etf(); idx = D["ret"].index; F = load_fred(idx); ratio = (F.VIX / F.VIX3M)
rv10 = vlib.rv(D["ret"]["SPY"], 10); vix = F.VIX / 100; spread = vix - rv10
d = tf("VIXY").reindex(idx); wd = pd.Series(idx.weekday, index=idx); notmon = (wd != 4).astype(float)
COST, BORROW = 1.5, 0.02
def leg(w, y="y_on", name="", p=True, shift=1):
    w = w.reindex(idx).fillna(0.0); r = d[y]
    g = w * r; cost = w.abs() * 2 * COST / 1e4 + w.clip(upper=0).abs() * BORROW / 252
    n = (g - cost).shift(shift).fillna(0.0).loc[START:DEV_END]; n[d[y].shift(shift).isna()] = 0.0
    if p: stats(n, name)
    return n
f95 = (ratio < 0.95).astype(float)
sd_on = d.y_on.rolling(20, min_periods=10).std()          # known at close t (last obs = y_on[t-1])
sd_on_l = sd_on.shift(1)                                    # careful: y_on[t] not known at close t -> use up to y_on[t-1]: rolling at t includes y_on[t]?? y_on[t] = close t -> 09:31 t+1, NOT known at t. So shift(1).
sd_cc = d.y_cc.rolling(20, min_periods=10).std().shift(1)
res = {}
print("== base")
res["r95"] = leg(-0.5 * notmon * f95, name="ratio<0.95 skip Fri (base)")
print("== vol scaling by trailing 20d sd of overnight VIXY returns (lagged), target sd 2% per night")
for tgt in [0.015, 0.02, 0.03]:
    w = -(tgt / sd_on_l).clip(upper=1.0) * notmon * f95
    res[f"r95_vs{tgt}"] = leg(w, name=f"ratio<0.95 vol-scaled tgt {tgt} (cap 1.0)")
    w = -(tgt / sd_on_l).clip(upper=0.5) * notmon * f95
    leg(w, name=f"   cap 0.5")
print("== scale by 1/VIX level")
for c in [0.08, 0.10]:
    w = -(c / vix).clip(upper=1.0) * notmon * f95; res[f"r95_ivix{c}"] = leg(w, name=f"ratio<0.95 w=-{c}/VIX (cap 1)")
print("== intraday leg: short VIXY 09:32 -> close (decided 09:31 of t) under conditions known at 09:31 of t")
# conditions known at close t-1: ratio.shift(1); at 09:31 t: VIXY gap zscore d.o_zgap (same day), SPY gap
rl = ratio.shift(1); vl = vix.shift(1)
def dleg(w, name, p=True): return leg(w, y="y_day", name=name, p=p, shift=0)
dleg(-0.5 + 0 * rl, "intraday short always")
for lo, hi in [(0.95, 1.0), (0.95, 1.05), (0.95, 9), (0.9, 1.05), (1.0, 9)]:
    dleg(-0.5 * ((rl >= lo) & (rl < hi)), f"intraday short if {lo}<=ratio(t-1)<{hi}")
for lo, hi in [(0.2, 0.25), (0.2, 0.3), (0.2, 0.35), (0.25, 0.35)]:
    dleg(-0.5 * ((vl >= lo) & (vl < hi)), f"intraday short if {lo}<=VIX(t-1)<{hi}")
gz = d.o_zgap
for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 2), (2, 9)]:
    dleg(-0.5 * ((gz >= lo) & (gz < hi)), f"intraday short if VIXY gap z in [{lo},{hi})")
for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 2), (2, 9)]:
    dleg(-0.5 * ((gz >= lo) & (gz < hi)) * ((rl >= 0.95)), f"   and ratio>=0.95")
spyg = tf("SPY").reindex(idx).o_zgap
for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 9)]:
    dleg(-0.5 * ((spyg >= lo) & (spyg < hi)), f"intraday short if SPY gap z in [{lo},{hi})")
print("== combos: overnight base + intraday flat-ratio leg")
res["day_r95_105"] = dleg(-0.5 * ((rl >= 0.95) & (rl < 1.05)), "intraday 0.95-1.05 alone")
res["combo"] = res["r95"] + res["day_r95_105"]; stats(res["combo"], "combo overnight r<0.95 + intraday 0.95-1.05")
print("== fold-OOS")
oos, tab = fold_oos(res, label="family"); print(tab.to_string())
for k, v in res.items(): fold_oos(v, label=k)
pd.DataFrame(res).to_csv("/home/user/llm_sharpe_optimization/agents/etfmacro/x5_res.csv")
