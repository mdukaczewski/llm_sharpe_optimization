#!/usr/bin/env python3
"""step 2: cost-aware selection on the saved ensemble: (a) a cap on the predicted round-trip spread cost f x (hs0931 + hs1001),
(b) a net score = calibrated expected gross (in-sample linear map of the ensemble rank to the mean traded return) minus the cost."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
log("==== step2: cost-aware selection on the saved ensemble ====")
S = pd.read_parquet(f"{OUT}/slim.parquet"); S = S[np.isfinite(S.ens)].reset_index(drop=True)
rows = []
# (a) cap on the round-trip spread cost
for f in [1.0, 0.5]:
    for cap in [5, 10, 15, 20, 30]:
        m = (f * S.hs_rt <= cap).to_numpy()
        for k in [4, 6, 8, 10]:
            L = positions(S, S.ens.to_numpy(), k=k, y="y_30d", mask=m); dd = daily(L, k)
            r = net(dd, f); s = stats(r); s.update(label=f"cap{cap}_f{f}_k{k}", f=f, cap=cap, k=k, kind="cap", days=len(dd), elig=m.mean(),
                                                   share_le300=(L["rank"] <= 300).mean(), share_gt600=(L["rank"] > 600).mean())
            rows.append(s); log(f"cap {cap:2d} f {f} k {k:2d} elig {m.mean():.2f} days {len(dd)} | Sh {s['sh']:5.2f} fo {s['fo']:5.2f} ret {s['ret']*100:6.1f}% mdd {s['mdd']*100:6.1f}% halves {s['h1']:5.2f}/{s['h2']:5.2f} | <=300 {s['share_le300']:.2f} >600 {s['share_gt600']:.2f}")
# (b) net score.  Calibration: in-sample OLS of the traded return y_30d (fill bin 3 -> exit bin 31) on the ensemble rank
# (per-day pct rank - 0.5, in [-0.5, 0.5]), over every (date, name) with a prediction: E[y] = a + b x rank.  Also the y_60d map.
for yc in ["y_30d", "y_60d"]:
    ok = S[yc].notna(); x = S.ens[ok].to_numpy(); yv = S[yc][ok].to_numpy()
    b, a = np.polyfit(x, yv, 1); log(f"calibration {yc}: E[y] = {a*1e4:+.2f} bps + {b*1e4:+.2f} bps x rank  (rank in [-0.5,0.5]; top name ~ {(a+0.5*b)*1e4:+.1f} bps, bottom ~ {(a-0.5*b)*1e4:+.1f} bps)")
    # also the non-linear check: mean y by decile of the rank and by the top/bottom 1% (the book trades the top ~0.5%)
    dec = pd.cut(S.ens[ok], [-0.5, -0.49, -0.45, -0.4, -0.3, -0.1, 0.1, 0.3, 0.4, 0.45, 0.49, 0.5], include_lowest=True)
    log(f"  mean {yc} (bps) by rank bucket: " + " ".join(f"{str(i.right):>5s}:{v*1e4:+.1f}" for i, v in S[yc][ok].groupby(dec, observed=True).mean().items()))
    if yc == "y_30d": A, B = a, b
E = A + B * S.ens.to_numpy()
for f in [1.0, 0.5]:
    c = f * S.hs_rt.to_numpy() / 1e4
    for k in [4, 6, 8, 10]:
        L = positions(S, None, k=k, y="y_30d", long_score=E - c, short_score=-E - c); dd = daily(L, k)
        r = net(dd, f); s = stats(r); s.update(label=f"netscore_f{f}_k{k}", f=f, cap=None, k=k, kind="netscore", days=len(dd), elig=1.0,
                                               share_le300=(L["rank"] <= 300).mean(), share_gt600=(L["rank"] > 600).mean())
        rows.append(s); log(f"netscore f {f} k {k:2d} days {len(dd)} | Sh {s['sh']:5.2f} fo {s['fo']:5.2f} ret {s['ret']*100:6.1f}% mdd {s['mdd']*100:6.1f}% halves {s['h1']:5.2f}/{s['h2']:5.2f} | <=300 {s['share_le300']:.2f} >600 {s['share_gt600']:.2f}")
        # net score at f, but costed at the other f too (sensitivity of the selection, not of the cost)
        for f2 in [1.0, 0.5, 0.25]:
            s2 = stats(net(dd, f2)); log(f"    selected at f={f}, costed at f={f2}: Sh {s2['sh']:5.2f} fo {s2['fo']:5.2f}")
        dd.to_parquet(f"{OUT}/step2_netscore_f{f}_k{k}.parquet")
pd.DataFrame(rows).to_csv(f"{OUT}/step2.csv", index=False); log("step2 done")
