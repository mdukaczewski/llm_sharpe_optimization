"""Exp 3: overnight vs intraday split of equity index ETFs (SPY QQQ IWM DIA) conditioned on regime variables known at the close.
ONE fixed rule set (written before looking at the tables below):
  overnight leg (MOC t -> MOO t+1):  +1 if VIX/VIX3M(t-1) < 1.0 and SPY 21d return (cc21) > 0, else 0
  intraday leg  (MOO t+1 -> MOC t+1): +1 if t+1 is in the turn-of-month window (last trading day .. 3rd trading day),
                                      -1 if VIX/VIX3M(t-1) >= 1.0 (and not TOM), else 0
equal risk across the 4 ETFs (0.05/rv20 lagged, cap 0.5); orders only where the position changes."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, elib2 as E
from rlib import sh
IDX = ["SPY", "QQQ", "IWM", "DIA"]
D = E.frame(IDX); D = D[~D.half.astype(bool)]
D["y_id"] = (1 + D.y_cc) / (1 + D.y_oo) - 1          # official open t+1 -> official close t+1 (dividend cancels)
Pv = lambda c: D.pivot(index="date", columns="tk", values=c)
YO, YI, RV = Pv("y_oo"), Pv("y_id"), Pv("rv20").shift(1)
idx = YO.index
vr = E.vix_ratio_lag(idx); tr = Pv("cc21")["SPY"]
dow = pd.Series(idx.weekday, idx)
# turn of month, from the trading calendar (known in advance); flag for the NEXT day's session
cal = pd.Series(idx, idx); ym = idx.to_period("M")
rk_f = cal.groupby(ym).cumcount() + 1; rk_l = cal.groupby(ym).cumcount(ascending=False) + 1
tom = ((rk_f <= 3) | (rk_l == 1)); tom_next = tom.shift(-1).fillna(False).astype(bool)
qe = (idx.month.isin([3, 6, 9, 12])) & (rk_l <= 3).values
E.log("=== x3 conditional means, equal-weight 4 index ETFs, bp (ON = MOC t->MOO t+1, ID = MOO->MOC t+1) [descriptive]")
on = YO.mean(1); idr = YI.mean(1)
def tab(name, key):
    g = pd.DataFrame({"on": on, "id": idr, "k": key}).dropna()
    t = g.groupby("k").agg(n=("on", "size"), on=("on", "mean"), id=("id", "mean"), sh_on=("on", sh), sh_id=("id", sh))
    t[["on", "id"]] *= 1e4; E.log(f"-- {name}\n" + t.round(2).to_string())
tab("all", pd.Series(0, idx))
tab("VIX/VIX3M(t-1) bucket", pd.cut(vr, [0, 0.85, 0.95, 1.0, 1.1, 9]))
tab("SPY cc21 > 0", tr > 0)
tab("dow of decision", dow)
tab("next session in TOM", tom_next)
tab("quarter-end last 3 days", pd.Series(qe, idx))
tab("half", pd.Series(np.where(idx < "2021-10-28", "h1", "h2"), idx))
# ---- the fixed rule set
a = ((vr < 1.0) & (tr > 0)).astype(float)                            # overnight leg
b = pd.Series(0.0, idx); b[tom_next] = 1.0; b[(vr >= 1.0) & ~tom_next] = -1.0   # intraday leg of t+1
base = (0.05 / RV).clip(upper=0.5).fillna(0)
def run(a, b, cost=1.5, borrow=0.01, name=""):
    A = base.mul(a, axis=0); B = base.mul(b, axis=0)     # A: held over night t (decided close t); B: held during session t+1 (decided close t)
    g = (A * YO.fillna(0)).sum(1) + (B * YI.fillna(0)).sum(1)
    Bprev = B.shift(1).fillna(0)                          # held during session t (i.e. into the close of t)
    dclose = (A - Bprev).abs(); dopen = (B - A).abs()
    c = (dclose.sum(1) + dopen.sum(1)) * cost / 1e4 + (A.clip(upper=0).abs().sum(1) + B.clip(upper=0).abs().sum(1)) * borrow / 252 / 2
    orders = (dclose > 1e-9).sum(1) + (dopen > 1e-9).sum(1)
    n = (g - c - orders * E.COMM_BPS / 1e4).loc[E.START:E.DEV_END]
    gross = (A.abs().sum(1) + B.abs().sum(1)) / 2
    E.report(name, n, orders.loc[n.index] / 2, gross.loc[n.index]); return n
E.log("=== x3 books (pos/day column = orders/2)")
res = {}
for cost in [1.5, 3.0]:
    res[f"split_rule_c{cost}"] = run(a, b, cost, name=f"split_rule_c{cost}")
    run(a, b * 0, cost, name=f"  on_leg_only_c{cost}")
    run(a * 0, b, cost, name=f"  id_leg_only_c{cost}")
    run(pd.Series(1.0, idx), pd.Series(0.0, idx), cost, name=f"  ref_on_always_c{cost}")
    run(pd.Series(1.0, idx), pd.Series(1.0, idx), cost, name=f"  ref_buyhold_c{cost}")
pd.DataFrame(res).to_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "x3_res.parquet"))
