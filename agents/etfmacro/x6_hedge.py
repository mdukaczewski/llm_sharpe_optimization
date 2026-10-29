"""r95 book: SPY-overnight hedge, SVXY-long implementation, post-spike overnight variants, correlations."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
D = load_etf(); idx = D["ret"].index; F = load_fred(idx); ratio = (F.VIX / F.VIX3M); vix = F.VIX / 100
d = tf("VIXY").reindex(idx); s = tf("SPY").reindex(idx); sv = tf("SVXY").reindex(idx)
wd = pd.Series(idx.weekday, index=idx); notmon = (wd != 4).astype(float); f95 = (ratio < 0.95).astype(float)
def leg(fr, w, y="y_on", cost=1.5, borrow=0.02, shift=1):
    w = w.reindex(idx).fillna(0.0); r = fr[y]
    g = w * r; c = w.abs() * 2 * cost / 1e4 + w.clip(upper=0).abs() * borrow / 252
    n = (g - c).shift(shift).fillna(0.0).loc[START:DEV_END]; n[fr[y].shift(shift).isna()] = 0.0; return n
base = leg(d, -0.5 * notmon * f95); stats(base, "base r95")
m = (notmon * f95 > 0)
x = s.y_on[m]; yv = d.y_on[m]; ok = x.notna() & yv.notna()
beta = np.cov(yv[ok], x[ok])[0, 1] / x[ok].var(); print("beta VIXY_on on SPY_on (in regime):", round(beta, 2), " corr", round(yv[ok].corr(x[ok]), 3))
for hb in [0.5, 1.0]:
    h = leg(s, -0.5 * notmon * f95 * (-beta) * hb * -1, cost=0.5, borrow=0.01)   # short SPY overnight: w = -0.5*|beta|*hb ... sign: short VIXY = long beta -> hedge = short SPY with weight 0.5*beta(neg)*? 
    # VIXY ~ beta*SPY with beta<0. position -0.5 VIXY ~ -0.5*beta SPY = +1.5 SPY. hedge = short 0.5*|beta|*hb SPY
    h = leg(s, 0.5 * beta * hb * notmon * f95, cost=0.5, borrow=0.01)
    stats(base + h, f"r95 hedged {hb*100:.0f}% (short {0.5*abs(beta)*hb:.2f} SPY overnight)")
print("-- SVXY long implementation (long 1.0 SVXY overnight; SVXY is -0.5x since 2018-02-28)")
sv_on = leg(sv, 1.0 * notmon * f95, cost=1.5, borrow=0.0); stats(sv_on, "long 1.0 SVXY overnight r95 skip Fri")
sv_on2 = leg(sv, 0.5 * notmon * f95, cost=1.5, borrow=0.0); stats(sv_on2.loc["2018-03-01":], "long 0.5 SVXY since 2018-03")
stats(base.loc["2018-03-01":], "base since 2018-03")
print("corr base vs SVXY impl:", round(base.corr(sv_on), 3))
print("-- post-spike overnight: short VIXY overnight when 5-day log mom of VIXY > 5% (existing ENS2 trigger, cc basis) / overnight basis")
c = D["close"]["VIXY"]; lv = np.log(c); mom5 = lv - lv.shift(5)
for thr in [0.05, 0.10]:
    stats(leg(d, -0.5 * (mom5 > thr)), f"short overnight if mom5>{thr} (all weekdays)")
    stats(leg(d, -0.5 * (mom5 > thr) * notmon), f"   skip Fri")
    stats(leg(d, -0.5 * (mom5 > thr) * notmon * (ratio >= 0.95)), f"   skip Fri, only ratio>=0.95 (complement of base)")
    stats(leg(d, -0.5 * (mom5 > thr), y="y_cc", borrow=0.02), f"   cc basis (close->close)")
stats(leg(d, -0.5 * notmon * ((vix >= 0.25) & (vix < 0.35))), "short overnight VIX in [25,35)")
stats(leg(d, -0.5 * notmon * (f95 + ((ratio >= 0.95) & (mom5 > 0.05)))), "base + post-spike complement (mom5>5%, ratio>=0.95)")
vv = (F.VIX.pct_change().rolling(20).std() * np.sqrt(252)); print("vol-of-vol quantiles", vv.quantile([.2, .5, .8]).round(2).to_dict())
for lo, hi in [(0, 0.8), (0.8, 1.1), (1.1, 1.5), (1.5, 9)]:
    stats(leg(d, -0.5 * notmon * ((vv >= lo) & (vv < hi))), f"short overnight if vol-of-vol in [{lo},{hi})")
    stats(leg(d, -0.5 * notmon * f95 * ((vv >= lo) & (vv < hi))), f"   and ratio<0.95")
print("-- correlations of base")
corr_report(base)
for nm, f in [("vol_ens2", "vol_ens2_net.csv"), ("onlgb1200q", "onlgb1200q_ls20.csv"), ("mom", "mom_longonly.csv"), ("daylgb", "daylgb1200_ls20.csv"), ("xslo20s1_150", "xslo20s1_150.csv")]:
    q = pd.read_csv(f"sleeves/{f}", index_col=0, parse_dates=True).iloc[:, 0].loc[:DEV_END]; print(nm, "Sh", round(sh(q), 2), "corr", round(base.reindex(q.index).fillna(0).corr(q), 3))
