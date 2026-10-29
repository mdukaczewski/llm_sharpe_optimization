"""Final-breadth runs: universe(1500). usage: exp5.py tag fs1:target:retrain:seed [fs2:...]  -> books + construction variants, series CSVs only."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
tag = sys.argv[1]; specs = sys.argv[2:]
NJ = int(os.environ.get("ONML_NJOBS", "2"))
D = load(1500); log(f"exp5 {tag}: {D.tk.nunique()} names")
F28 = FEATS18 + EXTRA10
SETS = {"f18": FEATS18, "f28": F28, "f28cal": F28 + CAL, "f28cal_nn": F28 + CAL + ["nnames"], "fall_nonn": F28 + XR + MR + CAL, "fall": F28 + XR + MR + CAL + ["nnames"]}
preds = {}; res = {}
def bk(nm, p, **kw):
    r = book2(D, p, p=False, **kw); res[nm] = r; save(f"{tag}_{nm}", r); log(report_line(f"{tag} {nm}", r)); return r
for sp in specs:
    fsn, tg, rt, s = sp.split(":"); s = int(s); t0 = time.time()
    p = wf(D, SETS[fsn], tg, rt, seed=s, n_jobs=NJ); preds[sp] = p; log(f"  trained {sp} {time.time()-t0:.0f}s  IC {rlib.ic(D, p)}")
    nm = f"{fsn}_{tg}_{rt}{s}"
    bk(f"{nm}_k20", p, k=20); bk(f"{nm}_k20p", p, k=20, weight="pred"); bk(f"{nm}_k30", p, k=30); bk(f"{nm}_k30p", p, k=30, weight="pred")
    bk(f"{nm}_k20_c15", p, k=20, cost=1.5); bk(f"{nm}_k20_even", p, k=20, half="even"); bk(f"{nm}_k20_odd", p, k=20, half="odd")
if len(specs) > 1:
    pe = sum(xrank(D, preds[sp]) for sp in specs) / len(specs)
    bk("ens_k20", pe, k=20); bk("ens_k20p", pe, k=20, weight="pred"); bk("ens_k30", pe, k=30); bk("ens_k20_c15", pe, k=20, cost=1.5)
rlib.fold_oos({k: v for k, v in res.items() if not any(x in k for x in ["c15", "even", "odd"])}, label=f"{tag} family")
log("exp5 done")
