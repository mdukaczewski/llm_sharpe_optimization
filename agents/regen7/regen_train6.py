"""train6.py continued into the holdout: usage regen_train6.py <cut> <f40|f26> <seed> [n_jobs=1]  (Q retrain, as in the dev run)."""
import os, sys, time
assert os.environ.get("STRAT_END") == "2026-10-28"
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml6")
from lib6 import *
from splice import splice, FIRST
cut = int(sys.argv[1]); fset = sys.argv[2]; seed = int(sys.argv[3]); nj = int(sys.argv[4]) if len(sys.argv) > 4 else 1
feats = {"f40": F40, "f26": F26X}[fset]; name = f"{fset}_c{cut}_Q_s{seed}"
LOGF = f"{LANE}/regen.log"
t0 = time.time(); D = model_frame(cut)
print(f"{name}: frame {len(D)} rows {D.date.min().date()}..{D.date.max().date()} {time.time()-t0:.0f}s", flush=True)
pn = xrank(D, wf(D, feats, "yr", "Q", seed=seed, n_jobs=nj, first=FIRST))
p = splice(D, pn, f"{OUT}/pred_{name}.parquet", f"onml6 {name} ({time.time()-t0:.0f}s)", LOGF, cast=np.float32)
save_pred(D, p, name)
print(f"{name} DONE {time.time()-t0:.0f}s", flush=True)
