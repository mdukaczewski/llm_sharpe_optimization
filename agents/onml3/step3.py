#!/usr/bin/env python3
"""Step 3: refit with liquidity / spread FEATURES (yearly retrain, seed 0, n_jobs=1).  Runs (name: features / target):
  R1 f26x_oo    : F26X          / yr_oo  (rank of y_oo, official-open exit)       -- no liq features, new target
  R2 f40_oo     : F26X + LIQF   / yr_oo
  R3 f40_oon    : F26X + LIQF   / yr_oon (rank of y_oo - MOO auction cost c=0.3 as a fraction)
  R4 f40_on     : F26X + LIQF   / yr     (rank of y_on, 09:31 exit, reference)
  R5 f26x_on    : F26X          / yr     (the old model, reference)
Predictions saved to out/pred_y_<name>.parquet; books: k=10 MOC+MOO c=0.3 (+ k sweep, cost caps 10/20).
usage: python3 step3.py R2 R1 R3 ..."""
from lib3 import *
NJ = int(os.environ.get("ONML_NJOBS", "1"))
from step3_runs import RUNS
which = [a for a in sys.argv[1:] if a in RUNS] or ["R2", "R1", "R3", "R4", "R5"]
t0 = time.time(); D = frame(2000); D["pc"] = (D.ce + D.cx) / 2
def books(D, p, name):
    G = D.copy(); G["p"] = p.values
    rows = []
    for k in [6, 8, 10, 12, 15]:
        cost_cols(G, k=k, c=0.3); G["pc"] = (G.ce + G.cx) / 2
        r = book3(G, k=k, y="y_oo", ce="ce", cx="cx"); m = metrics(r, k); log("  " + mline(f"{name} moo c0.3 k{k}", m)); rows.append((name, "moo", k, None, m))
        if k == 10:
            for cap in [10, 20]:
                r = book3(G, k=k, y="y_oo", ce="ce", cx="cx", cap=cap, capcol="pc"); m = metrics(r, k); log("  " + mline(f"{name} moo c0.3 k{k} cap{cap}", m)); rows.append((name, "moo", k, cap, m))
            G["c0"] = G.tk.map(rlib.cost_bps).astype(np.float32)
            r = book3(G, k=k, y="y_on", ce="c0", cx="c0"); m = metrics(r, k); log("  " + mline(f"{name} y_on rlib-cost k{k}", m)); rows.append((name, "on_rlib", k, None, m))
            G["hsx05"] = 0.5 * G.hs0931; r = book3(G, k=k, y="y_on", ce="ce", cx="hsx05"); m = metrics(r, k); log("  " + mline(f"{name} x05 k{k}", m)); rows.append((name, "x05", k, None, m))
            ic_oo = rlib.ic(G, G.p, "y_oo"); ic_on = rlib.ic(G, G.p, "y_on"); log(f"  {name} IC y_oo {ic_oo[0]:.4f} (t {ic_oo[1]:.1f}) | IC y_on {ic_on[0]:.4f} (t {ic_on[1]:.1f})")
    return rows
allrows = []
for rn in which:
    name, feats, target = RUNS[rn]; t1 = time.time()
    log(f"step3 {rn} {name}: {len(feats)} feats, target {target}, yearly seed0 n_jobs={NJ}")
    p = xrank(D, olib.wf(D, feats, target, "Y", seed=0, n_jobs=NJ)); save_pred(D, p, f"y_{name}")
    log(f"  fit {time.time()-t1:.0f}s")
    allrows += books(D, p, name)
T = pd.DataFrame([(a, b, k, c, round(m["sh_pre"], 3), round(m["sh"], 3), round(m["fold"], 3), round(m["h1"], 2), round(m["h2"], 2), round(m["mdd"], 3), round(m["ret"], 3)) for a, b, k, c, m in allrows],
                 columns=["model", "exec", "k", "cap", "sh_pre", "sh_04M", "fold_04M", "h1", "h2", "mdd", "ret"])
T.to_csv(f"{OUT}/step3_{'_'.join(which)}.csv", index=False); log("STEP3 DONE\n" + T.to_string(index=False))
