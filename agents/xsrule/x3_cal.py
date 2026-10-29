"""(d) overnight book vs its intraday complement; (e) calendar / FOMC conditioning of the all-names overnight premium."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule")
from xlib import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
D = load(N); D = rlib.clean(D, ["z30", "z60", "z5l", "zmoc", "zon_prev"])
FOMC = """2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29""".split()
FOMC = pd.DatetimeIndex(pd.to_datetime(FOMC))
dates = pd.DatetimeIndex(sorted(D.date.unique())); pos = pd.Series(np.arange(len(dates)), dates)
nxt = pd.Series(np.r_[dates[1:], [pd.NaT]], dates)          # next trading day after decision date t
# ---- (e) all-names equal weight overnight (cost 1-1.5 bp/side x2)
g = D[D.y_on.notna()].copy(); g["c"] = g.tk.map(cost_bps)
ew = g.groupby("date").agg(r=("y_on", "mean"), c=("c", "mean"), n=("tk", "size")); ew["net"] = ew.r - 2 * ew.c / 1e4
ew = ew.loc[START:DEV_END]
print(f"all-names EW overnight: n~{ew.n.mean():.0f}"); stats(ew.net, "EW overnight every day")
ew["dow_next"] = nxt.reindex(ew.index).dt.dayofweek.values   # weekday of the morning we exit
ew["fomc_next"] = nxt.reindex(ew.index).isin(FOMC).values     # exit morning is an FOMC day (pre-announcement drift)
ew["fomc_prev"] = ew.index.isin(FOMC)                          # decision day is the FOMC day (post-announcement overnight)
ew["mleft"] = D.groupby("date").mdays_left.first().reindex(ew.index)
ew["mpos"] = ew.groupby(ew.index.to_period("M")).cumcount()
print("by weekday of exit morning (Mon=0):"); print(ew.groupby("dow_next").net.agg(["mean", "std", "count"]).assign(sh=lambda x: x["mean"] / x["std"] * np.sqrt(252)).round(5))
print("by days left in month (0=last):"); print(ew[ew.mleft <= 4].groupby("mleft").net.agg(["mean", "count"]).assign(sh=lambda x: x["mean"] / ew.net.std() * np.sqrt(252)).round(5))
print("by position in month:"); print(ew[ew.mpos <= 3].groupby("mpos").net.agg(["mean", "count"]).assign(sh=lambda x: x["mean"] / ew.net.std() * np.sqrt(252)).round(5))
print("FOMC:"); print(ew.groupby(["fomc_next", "fomc_prev"]).net.agg(["mean", "count"]).assign(sh=lambda x: x["mean"] / ew.net.std() * np.sqrt(252)).round(5))
res = {}
res["ew_all"] = ew.net
res["ew_tom"] = ew.net.where((ew.mleft <= 1) | (ew.mpos <= 1), 0.0)        # turn of the month: last 2 + first 2 nights
res["ew_fomc"] = ew.net.where(ew.fomc_next, 0.0)
res["ew_notfri"] = ew.net.where(ew.dow_next != 0, 0.0)                     # skip the weekend night (Fri close -> Mon)
res["ew_tom_fomc"] = ew.net.where((ew.mleft <= 1) | (ew.mpos <= 1) | ew.fomc_next, 0.0)
for k, v in res.items(): summary(v, k)
# ---- (d) rule book overnight vs intraday complement
def legs(pred, k=20, smin=1.0):
    gg = D.loc[pred.notna(), ["date", "tk", "y_on", "y_id", "y_cc"]].copy(); gg["p"] = pred[gg.index].values; gg = gg[gg.y_on.notna()]
    gg["rk"] = gg.groupby("date").p.rank(method="first"); gg["n"] = gg.groupby("date").p.transform("size"); gg = gg[gg.n >= k + 5]
    top = gg[(gg.rk > gg.n - k) & (gg.p > smin)].copy(); top["c"] = top.tk.map(cost_bps) / 1e4; top["w"] = 1.0 / k
    a = top.groupby("date").agg(on=("y_on", lambda x: x.sum() / k), idl=("y_id", lambda x: x.fillna(0).sum() / k), cc=("y_cc", lambda x: x.fillna(0).sum() / k), c=("c", lambda x: x.sum() / k))
    a = a.reindex(dates).fillna(0.0).loc[START:DEV_END]; return a
base = 0.5 * D.z30 - D.z60; comp = base - 0.5 * D.z5l + 0.3 * D.zon_prev
for nm, s in [("base", base), ("comp", comp)]:
    a = legs(s)
    out = {f"{nm}_on": a.on - 2 * a.c, f"{nm}_id_long": a.idl - 2 * a.c, f"{nm}_id_short": -a.idl - 2 * a.c - a.c * 0 - (1.0 / 20) * 0,
           f"{nm}_cc_hold": a.cc - 2 * a.c, f"{nm}_on_minus_id": a.on - a.idl - 4 * a.c - 0.01 / 252 * 1.0}
    for k2, v in out.items(): summary(v, k2); res[k2] = v
    print(f"   corr on vs id leg {a.on.corr(a.idl):.3f}")
pd.DataFrame(res).to_parquet(f"{SCR}/x3_res_{N}.parquet")
