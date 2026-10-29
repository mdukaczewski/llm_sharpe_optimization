"""Phase 6: day-of-week structure in VIX ETPs, extreme-contango bins per year, ensemble of best sleeves with vol targeting."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]; idx = ret.index
F = pd.read_csv("/home/trumpet/strat20/agents/vol/fred_vix.csv", index_col=0, parse_dates=True).reindex(idx).ffill()
vix, vix3 = F["VIX"], F["VIX3M"]; ratio = vix / vix3
rv10 = rv(ret["SPY"], 10); vrv21 = rv(ret["VIXY"], 21)
r = ret.loc[START:END]
print("== VIXY/VIXM/SPY mean return by weekday (close-to-close, %/day) and Sharpe ==")
for t in ["VIXY", "VIXM", "UVXY", "SPY"]:
    g = r[t].groupby(r.index.weekday)
    print(f"{t:5s} " + " | ".join(f"{d}: {g.mean()[d]*100:+.3f}% Sh {sh(g.get_group(d)):+.2f} n={len(g.get_group(d))}" for d in range(5)))
# by year for VIXY Monday vs Friday
x = r["VIXY"]
print("VIXY by year: Mon / Tue / Wed / Thu / Fri mean %/day")
for y in range(2016, 2027):
    xy = x[str(y)]; print(f"  {y}: " + " ".join(f"{xy[xy.index.weekday==d].mean()*100:+.2f}" for d in range(5)))
# overnight (Fri close -> Mon open) and intraday Monday
co, oc = D["co"].loc[START:END], D["oc"].loc[START:END]
print("VIXY overnight by weekday (into that day): " + " ".join(f"{d}:{co['VIXY'][co.index.weekday==d].mean()*100:+.3f}%" for d in range(5)))
print("VIXY intraday  by weekday             : " + " ".join(f"{d}:{oc['VIXY'][oc.index.weekday==d].mean()*100:+.3f}%" for d in range(5)))
def run(name, w, fill="close", lag=1, full=False):
    W = pd.DataFrame(w, index=idx).fillna(0.0)
    return stats(backtest(W, ret, fill=fill, lag=lag, D=D), name, full=full)
print("\n== day-of-week sleeves (short VIXY held only over selected days; close fill, lag 1) ==")
wd = pd.Series(idx.weekday, index=idx)
# position decided at close t, filled close t+1, earns day t+2: to earn weekday d we need signal on day with weekday d-2 (business days)
for name, days in [("earn Mon only", [0]), ("earn Tue-Thu", [1, 2, 3]), ("earn Fri only", [4]), ("skip Mon", [1, 2, 3, 4]), ("skip Fri", [0, 1, 2, 3]), ("earn Thu+Fri", [3, 4])]:
    earn = wd.isin(days).astype(float)
    sig = earn.shift(-2).fillna(0)       # signal two trading days before the earning day
    run(f"short0.5 VIXY {name}", {"VIXY": -0.5 * sig})
print("\n== extreme contango bins, per-year stability (short VIXY if ratio<a, lag1) ==")
for a in [0.8, 0.83, 0.86, 0.9]:
    run(f"short0.5 VIXY if ratio<{a}", {"VIXY": -0.5 * (ratio < a).astype(float)}, full=True)
    run(f"long0.5 VIXY if ratio>{1.0}; short if ratio<{a}", {"VIXY": -0.5 * (ratio < a).astype(float) + 0.5 * (ratio > 1.0).astype(float)}, full=True)
print("\n== ensemble: inverse-vol weighted sleeves, then vol-target 15% ==")
sleeves = {}
f = (rv10 > 0.12).astype(float)
sleeves["spread_rv"] = backtest(pd.DataFrame({"VIXY": -0.5 * f, "VIXM": 0.5 * f}, index=idx), ret)["net"]
sleeves["short_contango"] = backtest(pd.DataFrame({"VIXY": -0.5 * (ratio < 0.83).astype(float)}, index=idx), ret)["net"]
sleeves["long_backwd"] = backtest(pd.DataFrame({"VIXY": 0.5 * (ratio > 1.05).astype(float)}, index=idx), ret)["net"]
sleeves["spread_always"] = backtest(pd.DataFrame({"VIXY": -0.5, "VIXM": 0.5}, index=idx), ret)["net"]
sleeves["short_mom5"] = backtest(pd.DataFrame({"VIXY": -0.5 * ((np.log(c["VIXY"]) - np.log(c["VIXY"]).shift(5)) > 0.05).astype(float)}, index=idx), ret)["net"]
S = pd.DataFrame(sleeves)
print(S.corr().round(2).to_string())
for nm in S: print(f"  {nm:16s} Sh {sh(S[nm]):5.2f}")
def ens(names, label):
    sub = S[names]; w = 1 / sub.std(); w /= w.sum(); e = (sub * w).sum(axis=1)
    eq = np.exp(np.log1p(e).cumsum()); dd = (eq / eq.cummax() - 1).min()
    yr = e.groupby(e.index.year).apply(sh)
    print(f"ENS {label:40s} Sh {sh(e):5.2f} ret {e.mean()*252*100:5.1f}% vol {e.std()*np.sqrt(252)*100:5.1f}% MDD {dd*100:6.1f}% wd {e.min()*100:6.1f}% ({e.idxmin().date()}) worstyr {yr.min():5.2f}({yr.idxmin()})")
    print("      yearly: " + " ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))
ens(["spread_rv", "short_contango"], "spread_rv + short_contango")
ens(["spread_rv", "short_contango", "long_backwd"], "+ long_backwd")
ens(["spread_rv", "short_contango", "long_backwd", "short_mom5"], "+ short_mom5")
ens(["spread_always", "short_contango", "long_backwd"], "spread_always + contango + backwd")
