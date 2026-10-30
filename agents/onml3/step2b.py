#!/usr/bin/env python3
"""Step 2b: extend the cost-aware selection on the saved predictions: lambda up to 6, tighter caps (2/3/4 bps),
per-tier calibration map E[y_oo | p-bin, tier] (in-sample), and cap x lambda combinations.  k in {6, 8, 10}."""
from lib3 import *
t0 = time.time(); D = frame(2000); P = load_pred(); G = D.merge(P, on=["date", "tk"], how="inner"); log(f"step2b: merged {len(G)} rows")
rows = []
def run(name, k, **kw):
    r = book3(G, k=k, **kw); m = metrics(r, k); log("  " + mline(name, m)); rows.append((name, k) + tuple(round(float(m[x]), 3) for x in ["sh_pre", "sh", "fold", "h1", "h2", "mdd", "ret"])); return r
G["pb"] = (G.p.clip(-0.5, 0.4999) + 0.5) * 50 // 1
cal = G.groupby("pb").y_oo.mean(); G["ey"] = G.pb.map(cal).astype(np.float32)
calt = G.groupby(["tier", "pb"]).y_oo.mean(); G["eyt"] = pd.MultiIndex.from_arrays([G.tier, G.pb]).map(calt).astype(np.float32)
log("  per-tier E[y_oo | p-bin] bps at bins 0 / 49: " + " | ".join(f"tier{t}: {calt[(t, 0)]*1e4:+.0f} / {calt[(t, 49)]*1e4:+.0f}" for t in range(4)))
log("--- tighter caps")
for k in [6, 8, 10]:
    cost_cols(G, k=k, c=0.3); G["pc"] = (G.ce + G.cx) / 2
    for cap in [2, 3, 4]:
        n_ok = (G.pc <= cap).groupby(G.date).sum().mean(); run(f"A_cap{cap}_k{k} (names/day {n_ok:.0f})", k, y="y_oo", ce="ce", cx="cx", cap=cap, capcol="pc")
log("--- lambda sweep, single map (ey) and per-tier map (eyt)")
for k in [6, 8, 10]:
    cost_cols(G, k=k, c=0.3); G["pc"] = (G.ce + G.cx) / 2
    for src in ["ey", "eyt"]:
        for lam in ([2.0, 3.0, 4.0, 6.0] if src == "ey" else [1.0, 2.0, 3.0, 4.0]):
            G["sL"] = G[src] - lam * (G.ce + G.cx) / 1e4; G["sS"] = -G[src] - lam * (G.ce + G.cx) / 1e4 - 0.01 / 252
            run(f"B_{src}_lam{lam}_k{k}", k, y="y_oo", ce="ce", cx="cx", score="sL", score_s="sS")
log("--- cap x lambda (single map)")
for k in [6, 8, 10]:
    cost_cols(G, k=k, c=0.3); G["pc"] = (G.ce + G.cx) / 2
    for cap in [5, 10]:
        for lam in [1.0, 2.0]:
            G["sL"] = G.ey - lam * (G.ce + G.cx) / 1e4; G["sS"] = -G.ey - lam * (G.ce + G.cx) / 1e4 - 0.01 / 252
            run(f"AB_cap{cap}_lam{lam}_k{k}", k, y="y_oo", ce="ce", cx="cx", score="sL", score_s="sS", cap=cap, capcol="pc")
log("--- c-sensitivity of the best step-2 books (selection fixed at the c=0.3 scores, costs re-evaluated at c)")
for k in [6, 8]:
    cost_cols(G, k=k, c=0.3); G["sL"] = G.ey - 2.0 * (G.ce + G.cx) / 1e4; G["sS"] = -G.ey - 2.0 * (G.ce + G.cx) / 1e4 - 0.01 / 252
    for c in [0.1, 0.3, 1.0]:
        G["ce2"], G["cx2"] = G.ce, G.cx; cost_cols(G, k=k, c=c); run(f"B_lam2_k{k} at c={c}", k, y="y_oo", ce="ce", cx="cx", score="sL", score_s="sS")
T = pd.DataFrame(rows, columns=["book", "k", "sh_pre", "sh_04M", "fold_04M", "h1", "h2", "mdd", "ret"]); T.to_csv(f"{OUT}/step2b.csv", index=False)
log("STEP2B DONE\n" + T.sort_values("fold_04M", ascending=False).head(12).to_string(index=False))
