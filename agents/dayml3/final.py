#!/usr/bin/env python3
"""dayml3 FINAL: quarterly retrain, n_jobs=2, of the chosen step-3 configurations (+ their pair ensemble), books costed at the
predicted half-spreads, candidates saved as dayml3_<desc>_x05_k{4,6,8} (f = 0.5) and dayml3_<desc>_x1_k{6,8} (f = 1).
  python3 agents/dayml3/final.py B1 B1x120          # the two models; 'ens' = average per-day pct rank of the two
Series saved: gross - f x spread cost - borrow, BEFORE the $3 commission (coordinator adds it from the _k count)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
NJ = int(os.environ.get("D3_NJOBS", "2")); FROM_SAVED = "--from-saved" in sys.argv
cfgs = [a for a in sys.argv[1:] if not a.startswith("--")]
DESC = {"A1": "liq", "B1": "liqnet", "C1": "liqnet600", "A0": "base", "B0": "net", "C0": "net600", "B1x120": "liqnet120", "C1x120": "liqnet600_120"}
SAVE = os.environ.get("D3_SAVE", "ens" if len(cfgs) > 1 else cfgs[0])        # which prediction to save candidates from
log(f"==== final: cfgs {cfgs} retrain Q n_jobs {NJ} save {SAVE} from_saved {FROM_SAVED} ====")
D = frame(1500); D["y_60d_net05"] = D["y_60d"] - 0.5 * D["hs_rt"] / 1e4; D["y_120d_net05"] = D["y_120d"] - 0.5 * D["hs_rt"] / 1e4
u600 = (D["rank"] <= 600).to_numpy(); u300 = (D["rank"] <= 300).to_numpy()
CFG = {"A0": (dlib.EXT, "y_60d", None), "A1": (EXTL, "y_60d", None), "B0": (dlib.EXT, "y_60d_net05", None), "B1": (EXTL, "y_60d_net05", None),
       "C0": (dlib.EXT, "y_60d_net05", u600), "C1": (EXTL, "y_60d_net05", u600), "B1x120": (EXTL, "y_120d_net05", None), "C1x120": (EXTL, "y_120d_net05", u600)}
preds = {}
for cfg in cfgs:
    feats, y, m = CFG[cfg]; fp = f"{OUT}/pred_final_{cfg}_Q.parquet"; t0 = time.time()
    if FROM_SAVED and os.path.exists(fp):
        P = pd.read_parquet(fp); pred = D[["date", "tk"]].merge(P, on=["date", "tk"], how="left")["p"].to_numpy(float)
    else:
        X = d2lib.matrix(D, feats); yr = d2lib.rank_target(D if m is None else D[m], y)
        if m is not None: yf = np.full(len(D), np.nan, np.float32); yf[m] = yr; yr = yf
        pred = d2lib.walk_forward(D, X, yr, retrain="Q", n_jobs=NJ, seed=0, verbose=True, train_mask=m); del X
        if m is not None: pred[~m] = np.nan
        P = D.loc[np.isfinite(pred), ["date", "tk"]].copy(); P["p"] = pred[np.isfinite(pred)]; P.to_parquet(fp)
    preds[cfg] = pred; icm, ict = rlib.ic(D, pd.Series(pred, index=D.index), "y_30d")
    log(f"final {cfg}: feats {len(feats)} target {y} fit {time.time()-t0:.0f}s IC(y_30d) {icm:.4f} t {ict:.1f}")
if len(cfgs) > 1:
    e = np.nanmean(np.column_stack([xr(D, preds[c]) for c in cfgs]), axis=1); e[~np.all(np.column_stack([np.isfinite(preds[c]) for c in cfgs]), axis=1)] = np.nan
    preds["ens"] = e
rows = []; books = {}
for tag, pred in preds.items():
    for uni, m in [("u1500", None), ("u600", u600), ("u300", u300)]:
        for k in [4, 6, 8]:
            L = positions(D, pred, k=k, y="y_30d", mask=m); dd = daily(L, k); books[(tag, uni, k)] = (L, dd)
            for f in [1.0, 0.5, 0.25]:
                s = stats(net(dd, f)); s.update(cfg=tag, uni=uni, k=k, f=f); rows.append(s)
            s = stats(net(dd, old=True)); s.update(cfg=tag, uni=uni, k=k, f="old"); rows.append(s)
            t = tier_table(L, 0.5); sh_ = " ".join(f"{r.tier}:{r.share:.2f}/{r.net_bps:+.1f}" for r in t.itertuples())
            log(f"{tag:7s} {uni:5s} k{k} | f1 Sh {rows[-4]['sh']:5.2f} fo {rows[-4]['fo']:5.2f} | f0.5 Sh {rows[-3]['sh']:5.2f} fo {rows[-3]['fo']:5.2f} h {rows[-3]['h1']:4.1f}/{rows[-3]['h2']:4.1f} mdd {rows[-3]['mdd']*100:5.0f}% | f0.25 fo {rows[-2]['fo']:5.2f} | old Sh {rows[-1]['sh']:5.2f} fo {rows[-1]['fo']:5.2f} | tiers {sh_}")
pd.DataFrame(rows).to_csv(f"{OUT}/final.csv", index=False)
# candidates: the SAVE prediction on the universe given by D3_UNI (default u1500)
uni = os.environ.get("D3_UNI", "u1500"); desc = DESC.get(SAVE, SAVE) if SAVE != "ens" else "_".join(DESC.get(c, c) for c in cfgs) + "_ens"
if uni != "u1500": desc += f"_{uni}"
ref = pd.read_csv(f"{R}/agents/dayml2/cands/dayml2_ens_e30_k8.csv", index_col=0, parse_dates=True).iloc[:, 0]
for k in [4, 6, 8]:
    L, dd = books[(SAVE, uni, k)]
    for f, tag in [(0.5, "x05"), (1.0, "x1")]:
        if tag == "x1" and k == 4: continue
        r = net(dd, f, comm=False); rc = net(dd, f); s = stats(rc); name = f"dayml3_{desc}_{tag}_k{k}"
        corr = pd.concat([rc, ref], axis=1).dropna().corr().iloc[0, 1]
        note = (f"intraday L/S top/bottom {k}, decide 09:31, fill 09:33 bar, exit 10:01 bar; model {SAVE} ({cfgs}, quarterly, EXT+liq feats); universe {uni}; "
                f"NET of f={f} x predicted NBBO half-spread (liq.py hs0931 entry + hs1001 exit) and borrow, BEFORE commission | at $0.4M: Sh {s['sh']:.2f} fold-OOS {s['fo']:.2f} halves {s['h1']:.2f}/{s['h2']:.2f} mdd {s['mdd']*100:.0f}% corr dayml2_ens_e30_k8 {corr:.2f}"
                + ("" if s['fo'] > 0 else " | NEGATIVE: not tradable at this spread assumption"))
        save(name, r, note); log(f"saved {name}: {note}")
log("final done")
