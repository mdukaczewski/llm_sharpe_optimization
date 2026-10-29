"""Stage D: feature ablations around 'fall' (yearly, seed 0). usage: exp4.py N tag"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
N = int(sys.argv[1]); tag = sys.argv[2]
tks = rlib.universe(1500)[:N]
D = load(tickers=tks)
F28 = FEATS18 + EXTRA10
SETS = {"fall_v2": F28 + XR + MR + CAL + ["nnames"], "f28cal": F28 + CAL, "fall_nonn": F28 + XR + MR + CAL, "f28cal_nn": F28 + CAL + ["nnames"], "f28_v2": F28}
preds = pd.DataFrame({"date": D.date, "tk": D.tk}); res = {}
for nm, fs in SETS.items():
    t0 = time.time(); p = wf(D, fs, "yr", "Y", seed=1 if nm == "fall_s1" else 0); preds[nm] = p
    r = rlib.book(D, p, k=20, p=False); res[nm] = r; save(f"{tag}_{nm}_yr", r); log(report_line(f"{tag} {nm} yr ({len(fs)}f) {time.time()-t0:.0f}s", r))
preds.to_parquet(f"{OUT}/preds_{tag}_abl.parquet")
log("exp4 done")
