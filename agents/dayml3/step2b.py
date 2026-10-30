#!/usr/bin/env python3
"""step 2b: net score with a NON-linear (bucketed) in-sample calibration of the ensemble rank to the mean y_30d:
40 buckets, 1%-wide at the extremes (|rank| > 0.45) and 5%-wide in the middle; E[y] = bucket mean."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
log("==== step2b: bucket-calibrated net score ====")
S = pd.read_parquet(f"{OUT}/slim.parquet"); S = S[np.isfinite(S.ens) & S.y_30d.notna()].reset_index(drop=True)
edges = np.r_[np.arange(-0.5, -0.45, 0.01), np.arange(-0.45, 0.45, 0.05), np.arange(0.45, 0.5001, 0.01)]
b = pd.cut(S.ens, edges, include_lowest=True); cal = S.y_30d.groupby(b, observed=True).mean()
log("bucket means (bps): " + " ".join(f"{i.right:.2f}:{v*1e4:+.1f}" for i, v in cal.items() if abs(i.right) > 0.44))
E = b.map(cal).astype(float).to_numpy()
rows = []
for f in [1.0, 0.5]:
    c = f * S.hs_rt.to_numpy() / 1e4
    for k in [4, 6, 8, 10]:
        L = positions(S, None, k=k, y="y_30d", long_score=E - c, short_score=-E - c); dd = daily(L, k)
        s = stats(net(dd, f)); s.update(label=f"netscore_bucket_f{f}_k{k}", f=f, k=k, share_le300=(L["rank"] <= 300).mean(), share_gt600=(L["rank"] > 600).mean()); rows.append(s)
        log(f"bucket netscore f {f} k {k:2d} | Sh {s['sh']:5.2f} fo {s['fo']:5.2f} ret {s['ret']*100:6.1f}% mdd {s['mdd']*100:6.1f}% halves {s['h1']:5.2f}/{s['h2']:5.2f} | <=300 {s['share_le300']:.2f} >600 {s['share_gt600']:.2f}")
pd.DataFrame(rows).to_csv(f"{OUT}/step2b.csv", index=False); log("step2b done")
