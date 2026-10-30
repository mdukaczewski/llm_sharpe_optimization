#!/usr/bin/env python3
"""Step 4: final run of the chosen configuration(s): quarterly retrain, 3 seeds rank-averaged, n_jobs=2.
Saves predictions out/pred_<desc>.parquet and candidates (series NET of auction cost c=0.3 + borrow, BEFORE commission):
  onml3_<desc>_moo_k{K}  (MOC entry + MOO exit), onml3_<desc>_x05_k10 / _x1_k10 (MOC entry at c=0.3, 09:31 exit paying 0.5 / 1 x hs0931).
usage: python3 step4.py <desc> [ks=6,8,10,12,15] [cap=none] [seeds=0,1,2]   desc in step3's RUNS (R1..R5 names)."""
from lib3 import *
from step3_runs import RUNS
NJ = int(os.environ.get("ONML_NJOBS", "2"))
desc = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
ks = [int(x) for x in kw.get("ks", "6,8,10,12,15").split(",")]; cap = None if kw.get("cap", "none") == "none" else float(kw["cap"]); seeds = [int(x) for x in kw.get("seeds", "0,1,2").split(",")]
feats, target = [(f, t) for n, f, t in RUNS.values() if n == desc][0]
t0 = time.time(); D = frame(2000); log(f"step4 FINAL {desc}: {len(feats)} feats target {target} quarterly seeds {seeds} n_jobs={NJ} cap={cap}")
members = []
for s in seeds:
    t1 = time.time(); members.append(xrank(D, olib.wf(D, feats, target, "Q", seed=s, n_jobs=NJ))); log(f"  quarterly seed {s} {time.time()-t1:.0f}s")
    save_pred(D, members[-1], f"{desc}_q_s{s}")
p = sum(members) / len(members); save_pred(D, p, desc)
G = D.copy(); G["p"] = p.values; G["c0"] = G.tk.map(rlib.cost_bps).astype(np.float32); G["hsx05"] = 0.5 * G.hs0931; G["hsx1"] = G.hs0931
capn = "" if cap is None else f"cap{int(cap)}_"
old = old_cand()
for k in ks:
    cost_cols(G, k=k, c=0.3); G["pc"] = (G.ce + G.cx) / 2
    r = book3(G, k=k, y="y_oo", ce="ce", cx="cx", cap=cap, capcol="pc"); m = metrics(r, k)
    name = f"onml3_{desc}_{capn}moo_k{k}"; log("  " + mline(f"FINAL {name}", m) + f" corr_old {corr_with(commission(r, k), old):.2f}")
    rlib.save_cand("onml3", name, r, f"3-seed quarterly LightGBM, {len(feats)} feats ({'with' if 'f40' in desc else 'no'} liq feats), target {target}, u2000, L/S top{k} eq, MOC entry + MOO exit, net of auction cost c=0.3 (order $400k/2k) + borrow 1%, before commission" + (f", cost cap {cap} bps/side" if cap else ""))
    for c in [0.1, 1.0]:
        cost_cols(G, k=k, c=c); r2 = book3(G, k=k, y="y_oo", ce="ce", cx="cx", cap=cap, capcol="pc"); m2 = metrics(r2, k); log("  " + mline(f"   c={c} {name}", m2))
    cost_cols(G, k=k, c=0.3)
    if k == 10:
        for f, col in [("x05", "hsx05"), ("x1", "hsx1")]:
            r = book3(G, k=k, y="y_on", ce="ce", cx=col, cap=cap, capcol="pc"); m = metrics(r, k); name = f"onml3_{desc}_{capn}{f}_k{k}"; log("  " + mline(f"FINAL {name}", m))
            rlib.save_cand("onml3", name, r, f"same predictions, MOC entry (auction c=0.3) + 09:31-bar exit paying {0.5 if f=='x05' else 1.0} x predicted half-spread hs0931, borrow 1%, before commission")
        r = book3(G, k=k, y="y_on", ce="c0", cx="c0"); m = metrics(r, k); log("  " + mline(f"   old-convention y_on rlib cost {desc} k10", m))
        r, L = book3(G, k=k, y="y_oo", ce="ce", cx="cx", cap=cap, capcol="pc", ret_pos=True); log(f"  tiers {desc} moo k10\n" + tier_table(L, "y_oo").round(2).to_string(index=False))
log(f"STEP4 DONE {desc} {time.time()-t0:.0f}s")
