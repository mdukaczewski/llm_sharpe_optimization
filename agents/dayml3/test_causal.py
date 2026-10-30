import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
D = frame(1500); u600 = (D["rank"] <= 600).to_numpy()
P = pd.read_parquet(f"{OUT}/pred_step3_A1_Y.parquet"); A1 = D[["date", "tk"]].merge(P, on=["date", "tk"], how="left")["p"].to_numpy(float)
log("==== causal calibration check (A1 yearly, cost-aware two-sided selection) ====")
for fsel in [0.5, 1.0]:
    ls, ss = sel_scores(D, A1, f=fsel)
    for uni, m in [("u1500", None), ("u600", u600)]:
        for k in [2, 3, 4, 6, 8]:
            L = positions(D, None, k=k, y="y_30d", mask=m, long_score=ls, short_score=ss); dd = daily(L, k)
            s = {f: stats(net(dd, f)) for f in [1.0, 0.5, 0.25]}; so = stats(net(dd, old=True))
            t = tier_table(L, 0.5); sh_ = " ".join(f"{r.tier}:{r.share:.2f}/{r.net_bps:+.1f}" for r in t.itertuples())
            log(f"A1sel{fsel}c {uni:5s} k{k} | f1 Sh {s[1.0]['sh']:5.2f} fo {s[1.0]['fo']:5.2f} | f0.5 Sh {s[0.5]['sh']:5.2f} fo {s[0.5]['fo']:5.2f} h {s[0.5]['h1']:4.1f}/{s[0.5]['h2']:4.1f} mdd {s[0.5]['mdd']*100:5.0f}% | f0.25 fo {s[0.25]['fo']:5.2f} | old Sh {so['sh']:5.2f} fo {so['fo']:5.2f} | tiers {sh_}")
