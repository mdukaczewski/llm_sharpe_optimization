"""Refinements: VIXM cc hedge overlay; UVXY/VXX instrument; entry bin; worst days."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
from ilib import Min
D = load_etf(); idx = D["ret"].index; F = load_fred(idx); ratio = (F.VIX / F.VIX3M); vix = F.VIX / 100
rv10 = vlib.rv(D["ret"]["SPY"], 10); vrp = vix - rv10
wd = pd.Series(idx.weekday, index=idx); notmon = (wd != 4).astype(float); f95 = (ratio < 0.95).astype(float)
reg = notmon * ((f95 > 0) | (vrp > 0.03))
d = tf("VIXY").reindex(idx)
def leg(fr, w, y="y_on", cost=1.5, borrow=0.02, shift=1):
    w = w.reindex(idx).fillna(0.0); r = fr[y]
    g = w * r; c = w.abs() * 2 * cost / 1e4 + w.clip(upper=0).abs() * borrow / 252
    n = (g - c).shift(shift).fillna(0.0).loc[START:DEV_END]; n[fr[y].shift(shift).isna()] = 0.0; return n
base = leg(d, -0.5 * notmon * f95); stats(base, "r95")
base2 = leg(d, -0.5 * reg); stats(base2, "r95|vrp3")
print("worst 8 days r95:"); print((base.sort_values().head(8) * 100).round(2).to_string())
print("worst 8 days r95|vrp3:"); print((base2.sort_values().head(8) * 100).round(2).to_string())
print("== VIXM close-to-close long overlay (daily, via vlib.backtest; decided close t, filled close t+1)")
stats(bt(pd.DataFrame({"VIXM": 0.5 + 0 * notmon}, index=idx), D)["net"], "long 0.5 VIXM always (cc)")
for wv in [0.25, 0.5]:
    for nm, f in [("r95", notmon * f95), ("r95|vrp3", reg)]:
        W = pd.DataFrame({"VIXM": wv * f.shift(-1).fillna(0)}, index=idx)     # vlib fills at t+1 close: shift so the VIXM leg is on when the overnight leg is on (held from close t+1 ... ) -- approximate overlay
        h = bt(W, D)["net"]; b = base if nm == "r95" else base2
        stats(b + h, f"{nm} + long {wv} VIXM cc overlay")
print("== instrument: UVXY (2x, borrow 5%) at half weight; VXX")
u = tf("UVXY").reindex(idx); x = tf("VXX").reindex(idx)
stats(leg(u, -0.25 * notmon * f95, borrow=0.05), "short 0.25 UVXY overnight r95")
stats(leg(x, -0.5 * notmon * f95, borrow=0.02), "short 0.5 VXX overnight r95")
stats(0.5 * base + 0.5 * leg(u, -0.25 * notmon * f95, borrow=0.05), "avg VIXY/UVXY")
print("== entry bin: decision at bin m, fill at close of bin m+1 (390 = official close); exit at 09:31 (bin 1)")
M = Min("VIXY", end=DEV_END); md = pd.DatetimeIndex(M.dates)
for m in [329, 359, 374, 384, 389]:
    pin = M.c[:, min(m + 1, 390)]; pout = np.r_[M.c[1:, 1], np.nan]
    r = pd.Series(pout / pin - 1, index=md).reindex(idx)
    r[np.abs(r) > 0.5] = np.nan
    w = (-0.5 * notmon * f95)
    n = (w * r - w.abs() * 2 * 1.5e-4 - w.abs() * 0.02 / 252).shift(1).fillna(0).loc[START:DEV_END]
    stats(n, f"entry fill bin {min(m+1,390)} r95")
print("== exit bin: entry at official close, exit fill at bin e")
for e in [1, 2, 5, 10, 15, 30]:
    pout = np.r_[M.c[1:, e], np.nan]; r = pd.Series(pout / M.close - 1, index=md).reindex(idx); r[np.abs(r) > 0.5] = np.nan
    w = (-0.5 * notmon * f95); n = (w * r - w.abs() * 2 * 1.5e-4 - w.abs() * 0.02 / 252).shift(1).fillna(0).loc[START:DEV_END]
    stats(n, f"exit fill bin {e} r95")
