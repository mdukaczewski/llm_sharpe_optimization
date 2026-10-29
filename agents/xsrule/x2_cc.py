"""(b)/(c) next-day close-to-close horizon: rule reversal/continuation books and a learned LGB L/S, hold 1/2/5."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule")
from xlib import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
RT = sys.argv[2] if len(sys.argv) > 2 else "Y"
t0 = time.time(); D = load(N); print(f"{time.time()-t0:.0f}s", flush=True)
EXTRA = ["z15l", "z5l", "zmoc", "zvwap", "vmoc_share", "vfirst_share", "cc21", "rv5_20", "zcc_l1", "z60_l1", "z30_l1", "zon_l1", "zon5", "dow", "mdays_left", "ldv", "spy_zgap"]
FE = rlib.FEATS18 + EXTRA
D = rlib.clean(D, [c for c in FE if c not in ("dow", "mdays_left", "ldv")])
res = {}
def run(tag, r): 
    res[tag] = r; st = stats(r, tag, p=False); oos, tab = fold_oos(r, p=False)
    print(f"{tag:30s} Sh {st['sharpe']:5.2f} OOS {sh(oos):5.2f} h {st['h1']:.2f}/{st['h2']:.2f} mdd {st['mdd']*100:5.1f}% minfold {tab.fold_sh.min():5.2f}", flush=True)
# --- rule books on y_cc
rules = {"rev_zcc": -D.zcc, "rev_cc5": -D.cc5, "rev_z60": -D.z60, "base": 0.5 * D.z30 - D.z60, "cont_zon": D.zon_prev,
         "rev_zcc_mr": -(D.zcc - D.groupby("date").zcc.transform("median")), "rev_zcc_z60": -D.zcc - 0.5 * D.z60,
         "rev_cc5_zcc": -D.cc5 - D.zcc, "rank_rev": -D.r_zcc - D.r_cc5 - 0.5 * D.r_z60, "rev_vwap": -D.zvwap - D.zcc}
for nm, s in rules.items():
    for hold in [1, 2, 5]:
        r, to = heldbook(D, s, k=20, mode="ls", hold=hold, name=f"cc_{nm}_ls20_h{hold}", p=False, turn=True)
        run(f"cc_{nm}_ls20_h{hold} to{to:.2f}", r)
    r = heldbook(D, s, k=20, mode="lo", hold=1, name=f"cc_{nm}_lo20", p=False); run(f"cc_{nm}_lo20_h1", r)
# --- learned
D["ycc_r"] = D.groupby("date").y_cc.rank(pct=True) - 0.5
D["ycc5_r"] = D.groupby("date").y_cc5.rank(pct=True) - 0.5
for feats, fnm in [(rlib.FEATS18, "f18"), (FE, "f35")]:
    t1 = time.time(); pred = rlib.walk_forward(D, feats, "ycc_r", retrain=RT, n_jobs=2, verbose=False, label_lag_days=3)
    print(f"  lgb {fnm} cc1 IC {rlib.ic(D, pred, 'y_cc')} {time.time()-t1:.0f}s", flush=True)
    pred.to_frame("p").to_parquet(f"{SCR}/x2_pred_{fnm}_{N}_{RT}.parquet")
    for k in [20, 30]:
        for hold in [1, 2, 5]:
            r, to = heldbook(D, pred, k=k, mode="ls", hold=hold, p=False, turn=True); run(f"cclgb_{fnm}_ls{k}_h{hold} to{to:.2f}", r)
    r = heldbook(D, pred, k=20, mode="lo", hold=1, p=False); run(f"cclgb_{fnm}_lo20_h1", r)
    # also: same prediction applied to the overnight-only and intraday-only legs (where does the c->c alpha live?)
    run(f"cclgb_{fnm}_ls20_onleg", rlib.book(D, pred, k=20, y="y_on", mode="ls", p=False))
    run(f"cclgb_{fnm}_ls20_idleg", rlib.book(D, pred, k=20, y="y_id", mode="ls", p=False))
# 5-day target model, held 5 days
t1 = time.time(); pred5 = rlib.walk_forward(D, FE, "ycc5_r", retrain=RT, n_jobs=2, verbose=False, label_lag_days=8)
print(f"  lgb f35 cc5 IC {rlib.ic(D, pred5, 'y_cc5')} {time.time()-t1:.0f}s", flush=True)
pred5.to_frame("p").to_parquet(f"{SCR}/x2_pred5_f35_{N}_{RT}.parquet")
for hold in [1, 2, 5, 10]:
    r, to = heldbook(D, pred5, k=20, mode="ls", hold=hold, p=False, turn=True); run(f"cclgb5_f35_ls20_h{hold} to{to:.2f}", r)
pd.DataFrame(res).to_parquet(f"{SCR}/x2_res_{N}_{RT}.parquet")
fold_oos({k: v for k, v in res.items() if k.startswith("cc_") and "_ls20_h1" in k}, label="cc rule LS h1 family")
fold_oos({k: v for k, v in res.items() if k.startswith("cclgb")}, label="cc lgb family")
