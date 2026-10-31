"""onml3 f40_on (R4: F26X + LIQF, target yr, quarterly, seeds 0,1,2, n_jobs 2) continued to 2026-10-28 + pred_f26calx_q3 re-merge."""
import os, sys, time
assert os.environ.get("STRAT_END") == "2026-10-28"
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml3")
from lib3 import *
from splice import splice, FIRST
LOGF = f"{rlib.HERE}/agents/onml6/regen.log"
D = frame(2000)
feats = F26X + LIQF
members = []
for s in [0, 1, 2]:
    t1 = time.time()
    pn = xrank(D, olib.wf(D, feats, "yr", "Q", seed=s, n_jobs=2, first=FIRST))
    m = splice(D, pn, f"{OUT}/pred_f40_on_q_s{s}.parquet", f"onml3 f40_on_q_s{s} ({time.time()-t1:.0f}s)", LOGF)
    members.append(m); save_pred(D, m, f"f40_on_q_s{s}")
p = sum(members) / len(members)
old = pd.read_parquet(f"{OUT}/pred_f40_on.parquet"); j = D[["date", "tk"]].assign(pn=p.values).merge(old, on=["date", "tk"], how="inner")
dev = j.date <= "2026-04-27"
with open(LOGF, "a") as fh: fh.write(f"onml3 pred_f40_on (3-seed mean): dev rows old {len(old)} matched {dev.sum()} maxabs {np.abs(j.pn[dev]-j.p[dev]).max():.3e}\n")
save_pred(D, p, "f40_on")
P = pd.read_parquet(PRED_OLD); q = D[["date", "tk"]].merge(P, on=["date", "tk"], how="left").p
old = pd.read_parquet(f"{OUT}/pred_f26calx_q3.parquet"); j = D[["date", "tk"]].assign(pn=q.values).merge(old, on=["date", "tk"], how="inner")
with open(LOGF, "a") as fh: fh.write(f"onml3 pred_f26calx_q3 (merge of agents/onml pred_f26calx_u2000_q3, max date {P.date.max().date()}): old rows {len(old)} matched {len(j)} maxabs {np.abs(j.pn-j.p).max():.3e}\n")
save_pred(D, q, "f26calx_q3")
print("ONML3 PREDS DONE", flush=True)
