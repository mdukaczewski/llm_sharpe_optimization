#!/usr/bin/env python3
"""Step 1: diagnosis on SAVED predictions (no fit).  Rank IC (per-day Spearman, within tier) and top/bottom-decile spread of
y_oo by tier and by year for p40, p26, pR; plus the onml3 yearly pooled vs liquid-only (pc10) models inside ranks 1-300."""
from lib5 import *
G = load_G(); G["pR"] = prank((G.p40 + G.p26) / 2, G.date)
for nm in ["y_f40_oo", "y_f40_oo_liq", "y_f40_on"]:
    P = pd.read_parquet(f"{HERE}/agents/onml3/out/pred_{nm}.parquet", columns=["date", "tk", "p"]).rename(columns={"p": nm})
    G = G.merge(P, on=["date", "tk"], how="left")
G["yr"] = G.date.dt.year
rows = []
for p in ["p40", "p26", "pR", "y_f40_on", "y_f40_oo", "y_f40_oo_liq"]:
    for y in ["y_oo"]:
        s = daily_ic(G, p, y); rows.append(dict(pred=p, tier="all(pooled)", ic=round(s.mean(), 4), t=round(s.mean() / s.std() * np.sqrt(len(s)), 1)))
        s = daily_ic(G, p, y, by="tier").reset_index()
        for t, x in s.groupby("tier"):
            v = x.iloc[:, -1]; rows.append(dict(pred=p, tier=TNAMES[t], ic=round(v.mean(), 4), t=round(v.mean() / v.std() * np.sqrt(len(v)), 1)))
T = pd.DataFrame(rows); tlog("STEP1 per-day rank IC vs y_oo (within tier = ranks recomputed inside the tier each day)", T.pivot(index="pred", columns="tier", values="ic").reset_index())
tlog("STEP1 t-stats", T.pivot(index="pred", columns="tier", values="t").reset_index())
# liquid-only model is defined only on the pc10 sub-universe: compare on the common rows in tier 0
C = G[(G.tier == 0) & G.y_f40_oo_liq.notna() & G.y_f40_oo.notna()]
for p in ["y_f40_oo", "y_f40_oo_liq", "p40", "pR"]:
    s = daily_ic(C, p, "y_oo"); log(f"STEP1 tier 1-300 common rows ({len(C)}): IC {p} {s.mean():.4f} t {s.mean()/s.std()*np.sqrt(len(s)):.1f}")
# by year, tier IC for pR and p40
yy = []
for p in ["pR", "p40", "p26"]:
    s = daily_ic(G, p, "y_oo", by="tier").reset_index(); s.columns = ["date", "tier", "ic"]; s["yr"] = s.date.dt.year
    t = s.groupby(["yr", "tier"]).ic.mean().unstack(); t.columns = [TNAMES[c] for c in t.columns]; t.insert(0, "pred", p); yy.append(t.reset_index())
tlog("STEP1 IC by year x tier", pd.concat(yy).round(4))
# decile spreads (bps of y_oo): deciles of the global rank (what the book uses) and within-tier deciles
def dec_spread(df, p, within):
    keys = ["date", "tier"] if within else ["date"]
    q = df.groupby(keys)[p].rank(pct=True); top = q > 0.9; bot = q <= 0.1
    out = {}
    for t in range(4):
        m = df.tier == t
        out[TNAMES[t]] = round((df.y_oo[m & top].mean() - df.y_oo[m & bot].mean()) * 1e4, 2)
    return out
dd = []
for p in ["pR", "p40", "p26"]:
    H = G[G[p].notna() & G.y_oo.notna()]
    dd.append(dict(pred=p, kind="global decile", **dec_spread(H, p, False)))
    dd.append(dict(pred=p, kind="within-tier decile", **dec_spread(H, p, True)))
    # extreme 1% (global): what the cost-aware book actually lives on
    q = H.groupby("date")[p].rank(pct=True)
    dd.append(dict(pred=p, kind="global top/bottom 2%", **{TNAMES[t]: round((H.y_oo[(H.tier == t) & (q > 0.98)].mean() - H.y_oo[(H.tier == t) & (q <= 0.02)].mean()) * 1e4, 2) for t in range(4)}))
tlog("STEP1 top-minus-bottom spread of y_oo, bps per name-night", pd.DataFrame(dd))
# by year for tier 0, within-tier decile spread, pR
H = G[G.pR.notna() & G.y_oo.notna() & (G.tier == 0)].copy(); q = H.groupby("date").pR.rank(pct=True)
H["d"] = np.where(q > 0.9, 1, np.where(q <= 0.1, -1, 0))
yr = H[H.d != 0].groupby(["yr", "d"]).y_oo.mean().unstack(); yr["spread_bps"] = (yr[1] - yr[-1]) * 1e4
log("STEP1 pR ranks 1-300 within-tier decile spread by year (bps):\n" + yr.spread_bps.round(2).to_string())
# share of rows / of the global extreme 2% by tier (how much the pooled loss "looks at" each tier)
q = G.groupby("date").pR.rank(pct=True); ext = (q > 0.98) | (q <= 0.02)
log("STEP1 share of rows by tier: " + " ".join(f"{TNAMES[t]} {(G.tier == t).mean():.3f}" for t in range(4))
    + " | share of global extreme-2% rows: " + " ".join(f"{TNAMES[t]} {(G.tier[ext] == t).mean():.3f}" for t in range(4)))
# cross-sectional dispersion of y_oo by tier (the target scale the rank loss ignores)
log("STEP1 per-day sd of y_oo by tier (bps, median over days): " + " ".join(f"{TNAMES[t]} {G[G.tier == t].groupby('date').y_oo.std().median()*1e4:.1f}" for t in range(4)))
log("STEP1 DONE")
