"""Stage B: portfolio construction variants on saved predictions. usage: exp2.py N tag predcol [predcol2 ...]"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
N = int(sys.argv[1]); tag = sys.argv[2]; cols = sys.argv[3:]
tks = rlib.universe(1500)[:N]
D = load(tickers=tks, extra=False)
P = pd.read_parquet(f"{OUT}/preds_{tag}.parquet")
D = D.merge(P, on=["date", "tk"], how="left")
for col in cols:
    p = D[col]; res = {}
    def run(nm, **kw):
        r = book2(D, p, p=False, **kw); res[nm] = r; log(report_line(f"{tag} {col} {nm}", r)); return r
    for k in [20, 30, 40]: run(f"k{k}_eq", k=k)
    run("k20_pred", k=20, weight="pred"); run("k30_pred", k=30, weight="pred")
    run("k20_vs", k=20, volscale=True); run("k30_vs", k=30, volscale=True)
    run("k40_vs", k=40, volscale=True)
    run("k30_q60", k=30, qmin=0.6); run("k40_q70", k=40, qmin=0.7)
    run("k20_even", k=20, half="even"); run("k20_odd", k=20, half="odd")
    run("k40_even", k=40, half="even"); run("k40_odd", k=40, half="odd")
    run("k20_c1.5", k=20, cost=1.5); run("k30_c1.5", k=30, cost=1.5)
    rlib.fold_oos({k_: v for k_, v in res.items() if "even" not in k_ and "odd" not in k_ and "c1.5" not in k_}, label=f"{tag} {col} construction family")
log("exp2 done")
