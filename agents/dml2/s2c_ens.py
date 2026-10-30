"""signal-set and weighting comparison on central band configs."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from s2_port import *
P, F, m = setup(); aud = liq_mats(P); sim = Sim(P, F, m, aud); res = []
CF = [(15, 5, 20), (15, 10, 20), (12, 1, 10), (20, 10, 45), (25, 5, 30)]
for tags in [["b1_r126", "b5_r126", "b10_r126"], ["b5_r126", "b10_r126"], ["b5_r126"], ["b10_r126"], ["s5_r126", "s10_r126"], ["b5_r126", "b10_r126", "s5_r126", "s10_r126"]]:
    S = score(P, m, tags, 5)
    for k, reb, buf in CF:
        for wm in (["eq", "iv"] if len(tags) == 2 and tags[0] == "b5_r126" else ["eq"]):
            df = sim.run(S, k=k, mode="band", reb=reb, buf=buf, wmode=wm); res.append(summ(df, f"{'+'.join(t[:-5] for t in tags)} k{k} reb{reb} buf{buf} {wm}"))
R = pd.DataFrame(res); R.to_csv(f"{OUT}/res_ens.csv", index=False)
print(R[["name", "sh", "foos", "gross", "vol", "mdd", "h1", "h2", "ord_d", "imp_bp", "com_bp", "gross_bp"]].to_string())
