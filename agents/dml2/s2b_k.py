"""plateau check in k (10/12/15/20) and buffer for the band book (b5+b10, sm5, hedged cap 0.5 and unhedged)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from s2_port import *
tags = sys.argv[1].split(","); P, F, m = setup(); aud = liq_mats(P); sim = Sim(P, F, m, aud); S = score(P, m, tags, 5); res = []
for k in [10, 12, 15, 20]:
    for reb in [1, 5, 10]:
        for buf in [10, 20, 30, 45]:
            for hg in [True, False]:
                df = sim.run(S, k=k, mode="band", reb=reb, buf=buf, hedge=hg); res.append(summ(df, f"k{k} reb{reb} buf{buf} {'H' if hg else 'U'}"))
R = pd.DataFrame(res); R.to_csv(f"{OUT}/res_kgrid_{'+'.join(tags)}.csv", index=False)
print(R[["name", "sh", "foos", "gross", "vol", "mdd", "h1", "h2", "ord_d", "imp_bp", "com_bp", "gross_bp"]].to_string())
