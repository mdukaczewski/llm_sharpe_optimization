"""(c)+(d) construction sweep at the commission on saved predictions.
usage: constr.py q3            -> agents/onml/out/pred_f28cal_q3.parquet (3-seed quarterly rank-avg; cut at DEV_END)
       constr.py raw:<name>    -> agents/onml2/out/raw_<name>.npy (yearly raw preds aligned to load(1500, extra=False))"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml2")
from o2lib import *
src = sys.argv[1]; tag = src.replace(":", "_")
D = load(1500, extra=False)
if src == "q3":
    Pq = pd.read_parquet(f"{HERE}/agents/onml/out/pred_f28cal_q3.parquet"); Pq = Pq[Pq.date <= rlib.DEV_END]
    D = D.merge(Pq, on=["date", "tk"], how="left"); p = D["p"]; raw = None
    log(f"constr q3: merged {p.notna().sum()} preds of {len(D)} rows, dates {Pq.date.min().date()}..{Pq.date.max().date()}")
else:
    raw = pd.Series(np.load(f"{OUT}/raw_{src[4:]}.npy"), index=D.index); p = xrank(D, raw)
res = {}
def run(nm, **kw):
    r, P = book3(D, p, raw=raw, **kw); res[nm] = ev(r, P, f"{tag} {nm}"); save(f"c_{tag}_{nm}", r, P); return r, P
for k in [6, 8, 10, 12, 15, 20, 30]:
    run(f"k{k}_eq", k=k); run(f"k{k}_pred", k=k, weight="pred")
for k in [10, 15, 20]: run(f"k{k}_lin", k=k, weight="lin")
if raw is not None:
    for k in [10, 15, 20]: run(f"k{k}_raw", k=k, weight="raw")
for k in [6, 10, 15, 20]: run(f"LO_k{k}_eq", k=k, mode="lo")
run("LO_k10_pred", k=10, mode="lo", weight="pred")
if raw is not None:
    for k, q in [(15, 0.6), (20, 0.5), (20, 0.7), (20, 0.8), (30, 0.8), (30, 0.9), (40, 0.9)]: run(f"k{k}_q{q}", k=k, qmin=q)
    for d in [0.3, 0.5]:
        for k in [10, 20]: run(f"k{k}_disp{d}", k=k, disp=d)
for k in [6, 10, 15, 20]: log(f"{tag} hold-through k{k} (top, bot) {hold_through(D, p, k)}")
tab = pd.DataFrame(res).T; tab.to_csv(f"{OUT}/constr_{tag}.csv", float_format="%.3f")
log(f"{tag} family fold-OOS at commission over k-sweep eq/pred: " + str(sh(rlib.fold_oos({n: with_comm(*load_out(f'c_{tag}_{n}')) for n in res if n.startswith('k')}, p=False)[0])))
log(f"constr {tag} done")
