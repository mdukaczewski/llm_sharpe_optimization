#!/usr/bin/env python3
"""step 3c: (a) the y_120d-style pair of the two-sided net models (B1x120 / S1x120: rank(+-y_120d - 0.5c)) and the 4-model
ensemble; (b) two-sided models trained on the net target at f = 1 (B1f1 / S1f1: rank(+-y_60d - 1.0 c)), to answer whether
anything is positive when crossing the spread."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
NJ = int(os.environ.get("D3_NJOBS", "1")); RETRAIN = "Y"
log(f"==== step3c: x120 pair + f=1 net targets, yearly, n_jobs {NJ} ====")
D = frame(1500)
D["yl120"] = D["y_120d"] - 0.5 * D["hs_rt"] / 1e4; D["ys120"] = -D["y_120d"] - 0.5 * D["hs_rt"] / 1e4
D["yl_f1"] = D["y_60d"] - 1.0 * D["hs_rt"] / 1e4; D["ys_f1"] = -D["y_60d"] - 1.0 * D["hs_rt"] / 1e4
u600 = (D["rank"] <= 600).to_numpy(); u300 = (D["rank"] <= 300).to_numpy()
def load(cfg):
    P = pd.read_parquet(f"{OUT}/pred_step3_{cfg}_{RETRAIN}.parquet"); return D[["date", "tk"]].merge(P, on=["date", "tk"], how="left")["p"].to_numpy(float)
def fit(cfg, feats, y, m=None):
    fp = f"{OUT}/pred_step3_{cfg}_{RETRAIN}.parquet"
    if os.path.exists(fp): return load(cfg)
    t0 = time.time(); X = d2lib.matrix(D, feats); yr = d2lib.rank_target(D, y)
    pred = d2lib.walk_forward(D, X, yr, retrain=RETRAIN, n_jobs=NJ, seed=0); del X
    P = D.loc[np.isfinite(pred), ["date", "tk"]].copy(); P["p"] = pred[np.isfinite(pred)]; P.to_parquet(fp)
    log(f"step3c {cfg}: fit {time.time()-t0:.0f}s"); return pred
rows = []
def books2(tag, ls, ss, unis, ks=(4, 6, 8)):
    for uni, m in unis:
        for k in ks:
            L = positions(D, None, k=k, y="y_30d", mask=m, long_score=ls, short_score=ss); dd = daily(L, k)
            for f in [1.0, 0.5, 0.25]:
                s = stats(net(dd, f)); s.update(cfg=tag, uni=uni, k=k, f=f); rows.append(s)
            s = stats(net(dd, old=True)); s.update(cfg=tag, uni=uni, k=k, f="old"); rows.append(s)
            t = tier_table(L, 0.5); sh_ = " ".join(f"{r.tier}:{r.share:.2f}/{r.net_bps:+.1f}" for r in t.itertuples())
            log(f"{tag:9s} {uni:5s} k{k} | f1 Sh {rows[-4]['sh']:5.2f} fo {rows[-4]['fo']:5.2f} | f0.5 Sh {rows[-3]['sh']:5.2f} fo {rows[-3]['fo']:5.2f} h {rows[-3]['h1']:4.1f}/{rows[-3]['h2']:4.1f} mdd {rows[-3]['mdd']*100:5.0f}% | f0.25 fo {rows[-2]['fo']:5.2f} | old Sh {rows[-1]['sh']:5.2f} fo {rows[-1]['fo']:5.2f} | tiers {sh_}")
            dd.to_parquet(f"{OUT}/step3c_{tag}_{uni}_k{k}.parquet")
    pd.DataFrame(rows).to_csv(f"{OUT}/step3c_Y.csv", index=False)
U = [("u1500", None), ("u600", u600)]
B1, S1 = load("B1"), load("S1")
books2("B1+S1_k23", B1, S1, U, ks=(2, 3))
BL, BS = fit("B1x120", EXTL, "yl120"), fit("S1x120", EXTL, "ys120")
books2("B1x120+S1x120", BL, BS, U)
books2("ens4", (xr(D, B1) + xr(D, BL)) / 2, (xr(D, S1) + xr(D, BS)) / 2, U, ks=(2, 3, 4, 6, 8))
FL, FS = fit("B1f1", EXTL, "yl_f1"), fit("S1f1", EXTL, "ys_f1")
books2("B1f1+S1f1", FL, FS, U)
log("step3c done")
