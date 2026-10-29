"""asymmetric books (long k, short ks) at the commission; usage: asym.py q3 | raw:<name>"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml2")
from o2lib import *
src = sys.argv[1]; tag = src.replace(":", "_")
D = load(1500, extra=False)
if src == "q3":
    Pq = pd.read_parquet(f"{HERE}/agents/onml/out/pred_f28cal_q3.parquet"); Pq = Pq[Pq.date <= rlib.DEV_END]
    D = D.merge(Pq, on=["date", "tk"], how="left"); p = D["p"]; raw = None
else:
    raw = pd.Series(np.load(f"{OUT}/raw_{src[4:]}.npy"), index=D.index); p = xrank(D, raw)
res = {}
for k, ks in [(10, 10), (10, 5), (15, 5), (15, 10), (20, 10), (20, 5), (12, 6), (8, 4), (15, 15), (20, 20), (12, 12)]:
    r, P = book3(D, p, k=k, ks=ks); res[f"L{k}S{ks}"] = ev(r, P, f"{tag} L{k} S{ks}"); save(f"a_{tag}_L{k}S{ks}", r, P)
for k, ks, sg in [(15, 5, 0.5), (20, 10, 0.5), (15, 10, 0.67), (10, 5, 0.5)]:
    r, P = book3(D, p, k=k, ks=ks, sgross=sg); res[f"L{k}S{ks}g{sg}"] = ev(r, P, f"{tag} L{k} S{ks} sgross {sg}"); save(f"a_{tag}_L{k}S{ks}g{sg}", r, P)
sp = olib.spy_cc()
for nm in res:
    r, P = load_out(f"a_{tag}_{nm}"); j = pd.concat([with_comm(r, P), sp], axis=1).dropna(); res[nm]["corr_spy"] = j.iloc[:, 0].corr(j.iloc[:, 1])
tab = pd.DataFrame(res).T; tab.to_csv(f"{OUT}/asym_{tag}.csv", float_format="%.3f"); log("\n" + tab.round(2).to_string())
log(f"asym {tag} done")
