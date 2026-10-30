#!/usr/bin/env python3
"""Step 2: cost-aware selection on the SAVED f26calx predictions (no refit), MOC+MOO at c=0.3 + borrow.
A. cost cap: keep only names whose predicted per-side auction cost (mean of MOC entry and MOO exit, c=0.3, order = $400k/2k)
   is <= cap, then L/S top-k by p.  Sweep cap in {5, 10, 20, none} x k in {6, 8, 10, 12, 15}.
B. calibrated net score: E[y_oo | p] from a 50-bin IN-SAMPLE map of the per-day rank p (one map over the whole dev span,
   stated as in-sample; the map is monotone and has no per-name information), long score = E - (ce+cx)/1e4,
   short score = -E - (ce+cx)/1e4 - borrow/252; long the top-k by the long score, short the top-k by the short score.
   Also a shrunk version with the cost scaled by lambda in {0.5, 1, 2}."""
from lib3 import *
t0 = time.time(); D = frame(2000); P = load_pred()
G = D.merge(P, on=["date", "tk"], how="inner"); log(f"step2: merged {len(G)} rows {time.time()-t0:.0f}s")
rows = []
def run(name, k, **kw):
    r = book3(G, k=k, **kw); m = metrics(r, k); log("  " + mline(name, m))
    rows.append((name, k) + tuple(round(float(m[x]), 3) for x in ["sh_pre", "sh", "fold", "h1", "h2", "mdd", "ret"])); return r
log("--- A. cost cap on the mean per-side auction cost (c=0.3), then top-k by p")
for k in [6, 8, 10, 12, 15]:
    cost_cols(G, k=k, c=0.3); G["pc"] = (G.ce + G.cx) / 2
    for cap in [5, 10, 20, None]:
        n_ok = (G.pc <= cap).groupby(G.date).sum().mean() if cap else G.groupby("date").size().mean()
        run(f"A_cap{cap}_k{k} (names/day {n_ok:.0f})", k, y="y_oo", ce="ce", cx="cx", cap=cap, capcol="pc")
log("--- B. calibrated net score (in-sample 50-bin map of E[y_oo | p])")
G["pb"] = (G.p.clip(-0.5, 0.4999) + 0.5) * 50 // 1
cal = G.groupby("pb").y_oo.mean(); log("  E[y_oo | p-bin] bps: " + " ".join(f"{int(b)}:{v*1e4:+.0f}" for b, v in cal.items() if int(b) in (0, 1, 2, 5, 10, 25, 40, 45, 47, 48, 49)))
G["ey"] = G.pb.map(cal).astype(np.float32)
for k in [6, 8, 10, 12, 15]:
    cost_cols(G, k=k, c=0.3)
    for lam in [0.5, 1.0, 2.0]:
        G["sL"] = G.ey - lam * (G.ce + G.cx) / 1e4; G["sS"] = -G.ey - lam * (G.ce + G.cx) / 1e4 - 0.01 / 252
        run(f"B_lam{lam}_k{k}", k, y="y_oo", ce="ce", cx="cx", score="sL", score_s="sS")
T = pd.DataFrame(rows, columns=["book", "k", "sh_pre", "sh_04M", "fold_04M", "h1", "h2", "mdd", "ret"]); T.to_csv(f"{OUT}/step2.csv", index=False)
log("STEP2 DONE\n" + T.sort_values("fold_04M", ascending=False).head(15).to_string(index=False))
