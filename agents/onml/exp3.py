"""Stage C: ensembles (seeds x retrain windows) at fixed breadth. usage: exp3.py N tag featset target"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
N = int(sys.argv[1]); tag = sys.argv[2]; fsn = sys.argv[3]; tg = sys.argv[4] if len(sys.argv) > 4 else "yr"
tks = rlib.universe(1500)[:N]
D = load(tickers=tks)
F28 = FEATS18 + EXTRA10
SETS = {"f18": FEATS18, "f28": F28, "f28xr": F28 + XR, "f28xrmr": F28 + XR + MR, "fall": F28 + XR + MR + CAL + ["nnames"]}
fs = SETS[fsn]
preds = pd.DataFrame({"date": D.date, "tk": D.tk}); res = {}
def bk(nm, p):
    r = rlib.book(D, p, k=20, p=False); res[nm] = r; save(f"{tag}_{nm}", r); log(report_line(f"{tag} {nm}", r)); return r
for s in [0, 1, 2]:
    t0 = time.time(); preds[f"Y_s{s}"] = wf(D, fs, tg, "Y", seed=s); bk(f"{fsn}_{tg}_Y_s{s}", preds[f"Y_s{s}"]); log(f"  {time.time()-t0:.0f}s")
pY = sum(preds[f"Y_s{s}"] for s in [0, 1, 2]) / 3; preds["Y_avg3"] = pY; bk(f"{fsn}_{tg}_Y_avg3", pY)
t0 = time.time(); preds["Q_s0"] = wf(D, fs, tg, "Q", seed=0); bk(f"{fsn}_{tg}_Q_s0", preds["Q_s0"]); log(f"  Q {time.time()-t0:.0f}s")
preds["YQ_avg"] = (pY + preds["Q_s0"]) / 2; bk(f"{fsn}_{tg}_YQ_avg", preds["YQ_avg"])
preds["YQ_avg2"] = (preds["Y_s0"] + preds["Q_s0"]) / 2; bk(f"{fsn}_{tg}_YQs0_avg", preds["YQ_avg2"])
preds.to_parquet(f"{OUT}/preds_{tag}_ens.parquet")
rlib.fold_oos(res, label=f"{tag} ensemble family")
log("exp3 done")
