"""Exp 2: pooled small LightGBM across equity/sector ETFs predicting the vol-scaled overnight return (MOC->MOO).
Yearly expanding retrain from 2018, label lag 3 days, n_jobs=1."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, elib2 as E, rlib
from rlib import sh
H = os.path.dirname(os.path.abspath(__file__))
univ = E.EQ if (len(sys.argv) < 2 or sys.argv[1] == "eq") else E.ALL; un = "eq" if univ is E.EQ else "all"
D = E.frame(univ); D = D[~D.half.astype(bool)].copy().sort_values(["date", "tk"]).reset_index(drop=True)
D["s"] = 0.5 * D.z30 - D.z60
D["rvl"] = D.groupby("tk").rv20.shift(1); D["rvd"] = D.rvl / np.sqrt(252)
for a, b in [("z30", "spy_z30"), ("z60", "spy_z60"), ("zoc", "spy_oc"), ("zgap", "spy_zgap")]: D[f"r_{a}"] = D[a] - D[b]
D["vixr"] = E.vix_ratio_lag(D.date).values
D["dow"] = D.date.dt.weekday
D["tkid"] = D.tk.map({t: i for i, t in enumerate(univ)})
F = ["z30", "z60", "z30l", "z120", "zoc", "zo15", "zgap", "rngpos", "rng", "zcc", "cc5", "rv20", "zon_prev", "spy_z60", "spy_z30", "spy_oc",
     "s", "r_z30", "r_z60", "r_zoc", "r_zgap", "vixr", "dow"]
if len(sys.argv) > 2 and sys.argv[2] == "nomkt":
    F = [f for f in F if not (f.startswith("spy_") or f in ("vixr", "dow"))]; un += "_nomkt"
# vlast_share / vday_rel dropped: they include the 16:00 auction bar (unknown to a MOC order)
D = rlib.clean(D, [f for f in F if f not in ("vixr", "dow", "rv20")])
D["yz"] = (D.y_oo / D.rvd).clip(-5, 5)
P = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0)
preds = {}
for seed in [0, 1, 2]:
    P["random_state"] = seed
    preds[seed] = rlib.walk_forward(D, F, "yz", retrain="Y", first="2018-01-01", params=P, n_jobs=1, verbose=False, label_lag_days=3)
D["pz"] = pd.concat(preds, axis=1).mean(1); D["pb"] = D.pz * D.rvd * 1e4   # predicted bps
D[["date", "tk", "pz", "pb", "y_oo", "y_on", "rvl", "s"]].to_parquet(f"{H}/x2_pred_{un}.parquet")
ok = D.pz.notna() & D.y_oo.notna()
E.log(f"=== x2 pooled LGB ({un}, {len(univ)} ETFs) rows {ok.sum()} | pooled corr(pz, yz) {D.loc[ok, ['pz','yz']].corr().iloc[0,1]:.3f}")
for y0 in range(2018, 2027):
    m = ok & (D.date.dt.year == y0); E.log(f"   {y0} corr {D.loc[m, ['pz','yz']].corr().iloc[0,1]:+.3f}  mean y_oo|pz>q80 {D.loc[m & (D.pz > D.loc[m].pz.quantile(.8)), 'y_oo'].mean()*1e4:6.2f} bp  |pz<q20 {D.loc[m & (D.pz < D.loc[m].pz.quantile(.2)), 'y_oo'].mean()*1e4:6.2f} bp")
q = pd.qcut(D.loc[ok, "pb"], 10, labels=False); E.log("   decile mean y_oo bp: " + " ".join(f"{v:.1f}" for v in D.loc[ok].groupby(q).y_oo.mean() * 1e4))
E.log("   decile mean pb  bp: " + " ".join(f"{v:.1f}" for v in D.loc[ok].groupby(q).pb.mean()))
Pv = lambda c: D[D.date >= "2018-01-01"].pivot(index="date", columns="tk", values=c)
PB, RV, Y, S = Pv("pb"), Pv("rvl"), Pv("y_oo"), Pv("s")
base = (0.05 / RV).clip(upper=0.5)
def cap(W, g=2.0):
    s_ = W.abs().sum(1); return W.mul(np.minimum(1.0, g / s_.replace(0, np.nan)).fillna(1.0), axis=0).fillna(0)
res = {}
for cost in [1.5, 3.0]:
    hur = 2 * cost + 1.0          # round trip cost + ~commission, bps
    books = {"hurdle_lo": base.where(PB > hur, 0.0),
             "hurdle_ls": base.where(PB > hur, 0.0) - base.where(PB < -(hur + 0.4), 0.0),
             "top3_lo": base.where(PB.rank(axis=1, ascending=False) <= 3, 0.0).where(PB > 0, 0.0),
             "s1_lo_ref": base.where(S > 1.0, 0.0)}
    for nm, W in books.items():
        n, g, npos, gr = E.run_book(cap(W), Y, cost); nm2 = f"ml_{un}_c{cost}_{nm}"; res[nm2] = n; E.report(nm2, n, npos, gr)
pd.DataFrame(res).to_parquet(f"{H}/x2_res_{un}.parquet")
import lightgbm as lgb
m = lgb.LGBMRegressor(**P, n_jobs=1, verbose=-1).fit(D.loc[D.yz.notna(), F], D.loc[D.yz.notna(), "yz"])
E.log("   gain importance: " + " ".join(f"{f}:{v:.0f}" for f, v in sorted(zip(F, m.booster_.feature_importance('gain')), key=lambda x: -x[1])[:12]))
