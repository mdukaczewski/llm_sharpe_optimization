"""Step 4 + robustness: hedge (cap 0.5 / 2 / none), smoothing, cost sensitivity on chosen configs.
usage: python s4_misc.py <tags> <k> <reb> <buf>"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from s2_port import *
tags = sys.argv[1].split(","); k, reb, buf = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
P, F, m = setup(); aud = liq_mats(P); sim = Sim(P, F, m, aud); res = []
spy = pd.Series(F["spy_ret"], P.dates)
def go(name, S, **kw):
    df = sim.run(S, k=k, mode="band", reb=reb, buf=buf, **kw); r = summ(df, name)
    j = pd.concat([df.net, spy], axis=1, join="inner"); r["corrSPY"] = round(j.corr().iloc[0, 1], 2)
    r["betaSPY_ols"] = round(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var(), 2); res.append(r); return df
base = f"{'+'.join(tags)} band k{k} reb{reb} buf{buf}"
for sm in [1, 5, 10]:
    S = score(P, m, tags, sm)
    go(f"{base} sm{sm} H0.5", S)
    if sm == 5:
        go(f"{base} sm{sm} H2.0", S, hedge_cap=2.0); go(f"{base} sm{sm} unhedged", S, hedge=False)
        go(f"{base} sm{sm} H0.5 c0.1", S, c=0.1); go(f"{base} sm{sm} H0.5 c1.0", S, c=1.0)
        go(f"{base} sm{sm} H0.5 B0(not tradable)", S, timing="B0")
        go(f"{base} sm{sm} H0.5 nocommission", S, order_usd=0)
        go(f"{base} sm{sm} H0.5 drift1.0", S, drift=1.0)
R = pd.DataFrame(res); R.to_csv(f"{OUT}/res_misc_{'+'.join(tags)}_k{k}r{reb}b{buf}.csv", index=False); print(R.to_string())
