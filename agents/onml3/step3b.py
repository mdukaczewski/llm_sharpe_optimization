#!/usr/bin/env python3
"""Step 3b: cost-aware selection (causal map, lambda 2/3) on the step-3 yearly predictions of each model (no refit).
usage: python3 step3b.py f40_oo f26x_oo ..."""
from lib3 import *
D = frame(2000); names = sys.argv[1:]; rows = []
for name in names:
    P = pd.read_parquet(f"{OUT}/pred_y_{name}.parquet"); G = D.merge(P, on=["date", "tk"], how="inner"); G["ey"] = causal_cal(G, first_year_insample=True)
    G["hsx05"] = 0.5 * G.hs0931
    for k in [6, 8, 10]:
        cost_cols(G, k=k, c=0.3)
        for lam in [2.0, 3.0]:
            net_scores(G, G.ey, lam); r = book3(G, k=k, y="y_oo", ce="ce", cx="cx", score="sL", score_s="sS"); m = metrics(r, k)
            log("  " + mline(f"{name} lam{lam:g} moo k{k}", m)); rows.append((name, "moo", k, lam, round(m["sh"], 3), round(m["fold"], 3), round(m["h1"], 2), round(m["h2"], 2), round(m["mdd"], 3)))
        if k in (6, 10):
            net_scores(G, G.ey, 3.0, cx="hsx05"); r = book3(G, k=k, y="y_on", ce="ce", cx="hsx05", score="sL", score_s="sS"); m = metrics(r, k)
            log("  " + mline(f"{name} lam3 x05 k{k}", m)); rows.append((name, "x05", k, 3.0, round(m["sh"], 3), round(m["fold"], 3), round(m["h1"], 2), round(m["h2"], 2), round(m["mdd"], 3)))
T = pd.DataFrame(rows, columns=["model", "exec", "k", "lam", "sh_04M", "fold_04M", "h1", "h2", "mdd"]); T.to_csv(f"{OUT}/step3b_{'_'.join(names)}.csv", index=False)
log("STEP3B DONE\n" + T.to_string(index=False))
