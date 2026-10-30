#!/usr/bin/env python3
"""dayml2 FINAL: regenerates ONLY the kept candidates end-to-end (universe(1500), honours STRAT_END via rlib.DEV_END).
  nohup python3 agents/dayml2/final.py > agents/dayml2/final.log 2>&1 &          (full refit, ~2 x 20-25 min with DAYML2_NJOBS=2)
  python3 agents/dayml2/final.py --from-saved                                      (books only, from agents/dayml/out/pred_y_{60d,120d}_Q.parquet)
Models: the two quarterly LightGBM walk-forwards of the dayml lane (dlib.EXT features, 300 trees / 15 leaves, expanding
window, label lag 3 days, rank target of y_60d resp. y_120d = return from the 09:33-bar fill (bin 3) to bin 61 / 121).
Books (decided at 09:31 of day t, filled at the 09:33 bar, 1 bp/side spread (1.5 beyond rank 600), borrow 1%/yr, equal weight):
  dayml2_x120q_e30_k6   y_120d model, long top-6 / short bottom-6, EXIT AT BIN 31 (10:01 bar)
  dayml2_ens_e30_k6/k8  average of the per-day pct ranks of the y_60d and y_120d models, exit bin 31, k = 6 / 8
  dayml2_ens_e60_k8     same ensemble, exit bin 61 (10:31 bar), k = 8
The saved series are spread-only net returns (rlib convention); the $3/order commission on $0.4M equity is
P x $6 / $400k per day (k6 L/S = 1.8 bps/day, k8 = 2.4 bps/day) and is reported alongside in REPORT.md."""
import sys, os, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml2")
from d2lib import *
LANE = "dayml2"; N = 1500; FROM_SAVED = "--from-saved" in sys.argv
KEPT = [  # name, model ('m60' | 'm120' | 'ens'), exit target, k, note
    ("dayml2_ens_e30_k8",  "ens",  "y_30d", 8, "avg rank of the quarterly y_60d+y_120d 09:31 LightGBMs, L/S top/bottom 8 eq, fill 09:33 bar (bin 3), exit 10:01 bar (bin 31)"),
    ("dayml2_ens_e30_k6",  "ens",  "y_30d", 6, "same ensemble, k=6, exit bin 31"),
    ("dayml2_x120q_e30_k6", "m120", "y_30d", 6, "quarterly y_120d model (dayml_x120dq) traded to the bin-31 exit, L/S 6 eq"),
    ("dayml2_ens_e60_k8",  "ens",  "y_60d", 8, "same ensemble, k=8, exit 10:31 bar (bin 61)"),
]
t0 = time.time(); log(f"final: DEV_END {rlib.DEV_END} n_jobs {NJ} from_saved {FROM_SAVED}")
tks = rlib.universe(N); D = build(tks, verbose=False, onml=False); log(f"final: rows {len(D)} names {D.tk.nunique()} {time.time()-t0:.0f}s")
preds = {}
for y, tag in [("y_60d", "m60"), ("y_120d", "m120")]:
    if FROM_SAVED: preds[tag] = load_preds(D, y)
    else:
        t1 = time.time(); X = matrix(D, dlib.EXT); yr = rank_target(D, y)
        preds[tag] = walk_forward(D, X, yr, retrain="Q", n_jobs=NJ, verbose=True); del X
        log(f"final: model {tag} fit {time.time()-t1:.0f}s IC {rlib.ic(D, pd.Series(preds[tag], index=D.index), y)}")
def xr(p): s = pd.Series(p, index=D.index); return (s.groupby(D.date).rank(pct=True) - 0.5).to_numpy()
preds["ens"] = (xr(preds["m60"]) + xr(preds["m120"])) / 2
out = {}
for name, model, y, k, note in KEPT:
    r, rn = book(D, preds[model], y, k=k, weight="eq"); d = ev(r, rn, name)
    rlib.save_cand(LANE, name, r, note + f" | $0.4M commission: Sh {d['sh_net']:.2f} fold-OOS {d['fo_net']:.2f}"); out[name] = (r, rn)
# report block: fold-OOS of the family, drawdown, correlations
fam_net = {n: v[1] for n, v in out.items()}; fam = {n: v[0] for n, v in out.items()}
oos, tab = rlib.fold_oos(fam_net, p=True, label="dayml2 family ($0.4M)"); rlib.fold_oos(fam, p=True, label="dayml2 family (spread)")
log("family picks:\n" + tab.to_string(index=False))
sl = pd.read_csv("/home/user/llm_sharpe_optimization/sleeves/onlgb1200q_ls20.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[:rlib.DEV_END]
spy = rlib.ticker_frame("SPY")["y_cc"].shift(1).loc[:rlib.DEV_END]   # close(t-1)->close(t), aligned to day t
base = rlib.load_cands("dayml").get("dayml_x60dq_k6")
for name, (r, rn) in out.items():
    eq = (1 + rn).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    c = pd.DataFrame({"x": rn, "on": sl, "spy": spy, "base": base}).dropna().corr()["x"]
    log(f"{name:22s} mdd($0.4M) {mdd*100:5.1f}% | corr onlgb1200q {c['on']:+.2f} SPY {c['spy']:+.2f} dayml_x60dq_k6 {c['base']:+.2f}")
log(f"FINAL DONE {time.time()-t0:.0f}s")
