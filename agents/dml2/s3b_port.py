"""step 3 portfolio check: yearly-retrain base vs +minute+onml features, same band configs."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from s2_port import *
P, F, m = setup(); aud = liq_mats(P); sim = Sim(P, F, m, aud); res = []
CF = [(15, 5, 20), (15, 10, 20), (12, 1, 10), (20, 10, 45), (25, 5, 30), (25, 1, 0)]
for lab, tags in [("Y_base", ["b5_Y_base", "b10_Y_base"]), ("Y_minon", ["b5_Y_minon", "b10_Y_minon"]), ("r126", ["b5_r126", "b10_r126"])]:
    S = score(P, m, tags, 5)
    for k, reb, buf in CF:
        df = sim.run(S, k=k, mode="band", reb=reb, buf=buf); res.append(summ(df, f"{lab} k{k} reb{reb} buf{buf}"))
    df = sim.run(S, k=25, mode="tranche", hold=5); res.append(summ(df, f"{lab} tranche k25 h5"))
R = pd.DataFrame(res); R.to_csv(f"{OUT}/res_s3.csv", index=False)
print(R[["name", "sh", "foos", "gross", "vol", "mdd", "h1", "h2", "ord_d", "imp_bp", "com_bp", "gross_bp"]].to_string())
