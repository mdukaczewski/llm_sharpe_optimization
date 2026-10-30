#!/usr/bin/env python3
"""Step 4: final run of the chosen configuration(s): quarterly retrain, 3 seeds rank-averaged, n_jobs=2.
Saves predictions out/pred_<desc>.parquet and candidates (series NET of the modelled execution cost + borrow, BEFORE commission):
  onml3_<desc>_lam<L>_moo_k{K}   MOC entry + MOO exit (auction cost c=0.3), cost-aware selection: long the top-k by
                                 E[gross | p] - L x (ce+cx), short the top-k by -E[gross | p] - L x (ce+cx) - borrow, where
                                 E[gross | p] is the causal expanding yearly 50-bin map of y_oo on the prediction rank (lib3.causal_cal)
  onml3_<desc>_lam<L>_x05_k10 / _x1_k10   MOC entry (auction c=0.3) + 09:31 exit paying 0.5 / 1 x hs0931; selection by the same
                                 rule with that exit cost.
usage: python3 step4.py <desc> [ks=6,8,10,12,15] [lam=3] [seeds=0,1,2] [skipfit=1]   desc = a name in step3_runs.RUNS."""
from lib3 import *
from step3_runs import RUNS, apply_filter
NJ = int(os.environ.get("ONML_NJOBS", "2"))
desc = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
ks = [int(x) for x in kw.get("ks", "6,8,10,12,15").split(",")]; lam = float(kw.get("lam", "3")); seeds = [int(x) for x in kw.get("seeds", "0,1,2").split(",")]
feats, target, filt = [(v[1], v[2], v[3] if len(v) > 3 else None) for v in RUNS.values() if v[0] == desc][0]
t0 = time.time(); D = apply_filter(frame(2000), filt); log(f"step4 FINAL {desc}: {len(feats)} feats target {target} quarterly seeds {seeds} n_jobs={NJ} lam={lam}")
if kw.get("skipfit") == "1" and os.path.exists(f"{OUT}/pred_{desc}.parquet"):
    P = pd.read_parquet(f"{OUT}/pred_{desc}.parquet"); p = D[["date", "tk"]].merge(P, on=["date", "tk"], how="left").p; log("  predictions loaded from disk")
else:
    members = []
    for s in seeds:
        t1 = time.time(); members.append(xrank(D, olib.wf(D, feats, target, "Q", seed=s, n_jobs=NJ))); log(f"  quarterly seed {s} {time.time()-t1:.0f}s")
        save_pred(D, members[-1], f"{desc}_q_s{s}")
    p = sum(members) / len(members); save_pred(D, p, desc)
G = D.copy(); G["p"] = p.values; G = G[G.p.notna()].reset_index(drop=True)
G["c0"] = G.tk.map(rlib.cost_bps).astype(np.float32); G["hsx05"] = 0.5 * G.hs0931; G["hsx1"] = G.hs0931
G["ey"] = causal_cal(G, first_year_insample=True)
old = old_cand(); tag = f"onml3_{desc}_lam{lam:g}"
ic_oo = rlib.ic(G, G.p, "y_oo"); ic_on = rlib.ic(G, G.p, "y_on"); log(f"  IC y_oo {ic_oo[0]:.4f} (t {ic_oo[1]:.1f}) | IC y_on {ic_on[0]:.4f} (t {ic_on[1]:.1f})")
def bk(k, y, ce, cx, **kw2): return book3(G, k=k, y=y, ce=ce, cx=cx, score="sL", score_s="sS", **kw2)
for k in ks:
    cost_cols(G, k=k, c=0.3); net_scores(G, G.ey, lam)
    r, L = bk(k, "y_oo", "ce", "cx", ret_pos=True); m = metrics(r, k); name = f"{tag}_moo_k{k}"
    log("  " + mline(f"FINAL {name}", m) + f" corr_old {corr_with(commission(r, k), old):.2f}")
    rlib.save_cand("onml3", name, r, f"3-seed quarterly LightGBM, {len(feats)} feats ({'with' if 'f40' in desc else 'no'} liq feats), target {target}, u2000, L/S {k}+{k} eq, MOC entry + MOO exit; cost-aware selection (E[gross|p] causal map - {lam:g} x auction cost); series net of auction cost c=0.3 (order $400k/{2*k}) + borrow 1%, before commission")
    for c in [0.1, 1.0]:
        cost_cols(G, k=k, c=c); m2 = metrics(bk(k, "y_oo", "ce", "cx"), k); log("  " + mline(f"   c={c} {name}", m2))
    cost_cols(G, k=k, c=0.3)
    r0 = book3(G, k=k, y="y_oo", ce="ce", cx="cx"); log("  " + mline(f"   plain top-k (no cost-aware sel) k{k}", metrics(r0, k)))
    if k == 10:
        log(f"  tiers {name}\n" + tier_table(L, "y_oo").round(2).to_string(index=False))
        for f, col in [("x05", "hsx05"), ("x1", "hsx1")]:
            net_scores(G, G.ey, lam, ce="ce", cx=col); r = bk(k, "y_on", "ce", col); m = metrics(r, k); name = f"{tag}_{f}_k{k}"; log("  " + mline(f"FINAL {name}", m))
            rlib.save_cand("onml3", name, r, f"same predictions, MOC entry (auction c=0.3) + 09:31-bar exit paying {0.5 if f=='x05' else 1.0} x predicted half-spread hs0931, cost-aware selection with that exit cost, borrow 1%, before commission")
        r = book3(G, k=k, y="y_on", ce="c0", cx="c0"); log("  " + mline(f"   old-convention plain top-10 y_on rlib cost {desc}", metrics(r, k)))
log(f"STEP4 DONE {desc} {time.time()-t0:.0f}s")
