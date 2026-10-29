"""pipeline check on 400 names: base yearly fit, construction sweep at the commission, hold-through."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml2")
from o2lib import *
D = load(400, extra=True); feats = F28CAL
t0 = time.time(); raw = wf2(D, feats, "yr", "Y", seed=0, n_jobs=2); log(f"t400 base yearly {time.time()-t0:.0f}s IC {ic(D, raw)}")
p = xrank(D, raw)
res = {}
for k in [6, 8, 10, 12, 15, 20, 30]:
    for w in ["eq", "pred"]:
        r, P = book3(D, p, k=k, weight=w); res[f"k{k}{w}"] = ev(r, P, f"t400 k{k} {w}")
for k in [10, 20]:
    r, P = book3(D, p, k=k, mode="lo"); ev(r, P, f"t400 LO k{k}")
for k, q in [(20, 0.5), (20, 0.7), (30, 0.7), (30, 0.8), (40, 0.8)]:
    r, P = book3(D, p, k=k, qmin=q, raw=raw); ev(r, P, f"t400 k{k} qmin{q}")
for d in [0.3, 0.5]:
    r, P = book3(D, p, k=20, disp=d, raw=raw); ev(r, P, f"t400 k20 disp>{d}")
for k in [10, 20]: log(f"t400 hold-through k{k} (top, bot) {hold_through(D, p, k)}")
log("t400 done")
