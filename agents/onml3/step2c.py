#!/usr/bin/env python3
"""Step 2c: the lambda selection of step 2 with a CAUSAL (expanding yearly) calibration map, on the saved predictions;
also applied to the spread-exit cases (score uses ce + f x hs0931)."""
from lib3 import *
D = frame(2000); P = load_pred(); G = D.merge(P, on=["date", "tk"], how="inner"); log(f"step2c: merged {len(G)} rows")
rows = []
def run(name, k, **kw):
    r = book3(G, k=k, **kw); m = metrics(r, k); log("  " + mline(name, m)); rows.append((name, k) + tuple(round(float(m[x]), 3) for x in ["sh_pre", "sh", "fold", "h1", "h2", "mdd", "ret"])); return r
G["ey_is"] = causal_cal(G, first_year_insample=True); G["ey_c"] = causal_cal(G, first_year_insample=False)
G["ey_full"] = G.groupby((G.p.clip(-0.5, 0.4999) + 0.5) * 50 // 1).y_oo.transform("mean").astype(np.float32)
log(f"  causal map: corr with full-sample map {G[['ey_is','ey_full']].dropna().corr().iloc[0,1]:.3f}; rows with map {G.ey_c.notna().mean()*100:.0f}% (strict) / {G.ey_is.notna().mean()*100:.0f}% (2017 in-sample)")
G["hsx05"] = 0.5 * G.hs0931; G["hsx1"] = G.hs0931
for k in [6, 8, 10, 12, 15]:
    cost_cols(G, k=k, c=0.3)
    for lam in [2.0, 3.0]:
        net_scores(G, G.ey_is, lam); run(f"C_moo_causal_lam{lam}_k{k}", k, y="y_oo", ce="ce", cx="cx", score="sL", score_s="sS")
        if k in (6, 10): net_scores(G, G.ey_c, lam); run(f"C_moo_strict_lam{lam}_k{k}", k, y="y_oo", ce="ce", cx="cx", score="sL", score_s="sS")
    if k in (6, 10):
        for f, col in [("x05", "hsx05"), ("x1", "hsx1")]:
            net_scores(G, G.ey_is, 2.0, ce="ce", cx=col); run(f"C_{f}_causal_lam2_k{k}", k, y="y_on", ce="ce", cx=col, score="sL", score_s="sS")
log("--- c-sensitivity of C_moo_causal_lam2 k6/k8/k10 (selection at c=0.3 costs, re-costed)")
for k in [6, 8, 10]:
    cost_cols(G, k=k, c=0.3); net_scores(G, G.ey_is, 2.0)
    for c in [0.1, 1.0]:
        cost_cols(G, k=k, c=c); run(f"C_moo_causal_lam2_k{k} at c={c}", k, y="y_oo", ce="ce", cx="cx", score="sL", score_s="sS")
T = pd.DataFrame(rows, columns=["book", "k", "sh_pre", "sh_04M", "fold_04M", "h1", "h2", "mdd", "ret"]); T.to_csv(f"{OUT}/step2c.csv", index=False)
log("STEP2C DONE\n" + T.to_string(index=False))
