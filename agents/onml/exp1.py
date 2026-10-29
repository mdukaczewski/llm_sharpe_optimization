"""Stage A: feature sets x targets at fixed breadth (yearly retrain). usage: exp1.py N tag"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
N = int(sys.argv[1]); tag = sys.argv[2]
tks = rlib.universe(1500)[:N]
D = load(tickers=tks); log(f"exp1 {tag}: {D.tk.nunique()} names")
F28 = FEATS18 + EXTRA10
SETS = {"f18": FEATS18, "f28": F28, "f28xr": F28 + XR, "f28xrmr": F28 + XR + MR, "fall": F28 + XR + MR + CAL + ["nnames"]}
preds = pd.DataFrame({"date": D.date, "tk": D.tk})
res = {}
for nm, fs in SETS.items():
    t0 = time.time(); p = wf(D, fs, "yr", "Y"); preds[f"{nm}_yr"] = p
    r = rlib.book(D, p, k=20, p=False); res[f"{nm}_yr"] = r; save(f"{tag}_{nm}_yr", r)
    log(report_line(f"{tag} {nm} yr ({len(fs)}f) {time.time()-t0:.0f}s", r))
best = max(res, key=lambda k: sh(rlib.fold_oos({"x": res[k]}, p=False)[0])); bf = best.split("_yr")[0]; fs = SETS[bf]; log("best feature set by foldOOS:", bf)
for tg in ["yv", "ycls"]:
    t0 = time.time(); p = wf(D, fs, tg, "Y"); preds[f"{bf}_{tg}"] = p
    r = rlib.book(D, p, k=20, p=False); res[f"{bf}_{tg}"] = r; save(f"{tag}_{bf}_{tg}", r)
    log(report_line(f"{tag} {bf} {tg} {time.time()-t0:.0f}s", r))
# average of rank-normalised predictions across targets
pavg = (xrank(D, preds[f"{bf}_yr"]) + xrank(D, preds[f"{bf}_yv"]) + xrank(D, preds[f"{bf}_ycls"])) / 3; preds[f"{bf}_avg3"] = pavg
r = rlib.book(D, pavg, k=20, p=False); res[f"{bf}_avg3"] = r; save(f"{tag}_{bf}_avg3", r); log(report_line(f"{tag} {bf} avg3 targets", r))
pavg2 = (xrank(D, preds[f"{bf}_yr"]) + xrank(D, preds[f"{bf}_yv"])) / 2; preds[f"{bf}_avg2"] = pavg2
r = rlib.book(D, pavg2, k=20, p=False); res[f"{bf}_avg2"] = r; save(f"{tag}_{bf}_avg2", r); log(report_line(f"{tag} {bf} avg yr+yv", r))
# 10:31 exit on same model (yr60 target) vs 09:31
p60 = wf(D, fs, "yr60", "Y"); preds[f"{bf}_yr60"] = p60
r = rlib.book(D, p60, k=20, y="y_on60", p=False); res[f"{bf}_yr60_x60"] = r; save(f"{tag}_{bf}_yr60_x60", r); log(report_line(f"{tag} {bf} yr60 exit10:31", r))
r = rlib.book(D, preds[f"{bf}_yr"], k=20, y="y_on60", p=False); log(report_line(f"{tag} {bf} yr model, exit10:31", r))
preds.to_parquet(f"{OUT}/preds_{tag}.parquet")
rlib.fold_oos(res, label=f"{tag} family")
log("exp1 done")
