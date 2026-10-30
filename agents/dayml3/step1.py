#!/usr/bin/env python3
"""step 1: the saved dayml2 ensemble books (k=8, k=6, exit bin 31) costed with the predicted NBBO half-spreads,
full universe and restricted to ranks <= 300 / <= 600."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
log("==== step1: saved predictions costed at the predicted half-spreads ====")
if os.path.exists(f"{OUT}/slim.parquet"): slim = pd.read_parquet(f"{OUT}/slim.parquet")
else:
    D = build(1500)
    slim = D[["date", "tk", "rank", "y_30d", "y_60d", "y_120d", "hs0931", "hs1001", "hs_rt", "cost_old"]].copy()
    slim["p60"] = load_saved(D, "y_60d"); slim["p120"] = load_saved(D, "y_120d"); slim["ens"] = ens_saved(D)
    slim.to_parquet(f"{OUT}/slim.parquet"); del D
S = slim; log(f"slim rows {len(S)} ens coverage {np.isfinite(S.ens).mean():.3f}")
hs = S[np.isfinite(S.ens)]; log("median hs0931 / hs1001 by tier (all rows): " + str(hs.groupby(tier(hs['rank']))[["hs0931", "hs1001"]].median().round(1).to_dict()))
rows = []; tiers = []
for uni, m in [("u1500", None), ("u600", (S["rank"] <= 600).to_numpy()), ("u300", (S["rank"] <= 300).to_numpy())]:
    for k in [8, 6]:
        L = positions(S, S.ens.to_numpy(), k=k, y="y_30d", mask=m); dd = daily(L, k)
        lab = f"ens_e30_k{k}_{uni}"; rows += ev(dd, lab)
        for f in [1.0, 0.5]:
            t = tier_table(L, f); t["book"] = lab; t["f"] = f; tiers.append(t)
            log(f"  tiers {lab} f={f}:\n" + t.round(2).to_string(index=False))
        dd.to_parquet(f"{OUT}/step1_{lab}.parquet")
# the tier-restricted books (select within the tier) at k=6 and k=8, for the tier breakdown of the full book
for lo, hi in TIERS:
    m = ((S["rank"] >= lo) & (S["rank"] <= hi)).to_numpy()
    for k in [8, 6]:
        L = positions(S, S.ens.to_numpy(), k=k, y="y_30d", mask=m); dd = daily(L, k); rows += ev(dd, f"ens_e30_k{k}_tier{lo}-{hi}", fs=(1.0, 0.5))
pd.DataFrame(rows).to_csv(f"{OUT}/step1.csv", index=False); pd.concat(tiers).to_csv(f"{OUT}/step1_tiers.csv", index=False)
log("step1 done")
