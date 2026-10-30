#!/usr/bin/env python3
"""Step 1: edge decay by exit time and tier.  Positions = the onml3 cost-aware k6 book (causal 50-bin map of y_oo,
lam 3, MOC + MOO cost at c = 0.3, borrow), per prediction set.  Gross bps per trade (side x return from the official close
of t) to the official open, 09:31, 09:35, 10:01, 10:31 bars of t+1, by rank tier and period; plus the extreme 5% of
each day's prediction rank (all names, a larger sample of the same decay) and the per-tier half-spreads at each time."""
from lib4 import *
G = load_G(); log(f"step1: G {len(G)} rows")
YS = ["y_oo", "y_on", "y_0935", "y_1001", "y_1031"]; LAB = ["open", "0931", "0935", "1001", "1031"]
G = G[G.y_oo.notna()].reset_index(drop=True)

def decay_tab(L, label):
    rows = []
    for i, nm in [(None, "all")] + list(enumerate(TNAMES)):
        x = L if i is None else L[L.tier == i]
        if len(x) == 0: continue
        sg = np.sign(x.w.values); r = dict(set=label, tier=nm, n=len(x), pos_pct=round(len(x) / len(L) * 100, 1))
        for y, lb in zip(YS, LAB): r[lb] = round(np.nanmean(sg * x[y].values) * 1e4, 2)
        for hs in ["n_hs0931", "n_hs0935", "n_hs1001"]: r["hs" + hs[-4:]] = round(np.nanmean(x[hs].values), 1)
        r["moo_c03"] = round(np.nanmean(x["cx"].values), 1)
        rows.append(r)
    return pd.DataFrame(rows)

out = []
for pc in ["p40", "p26"]:
    H = G[G[pc].notna()].copy(); H["p"] = H[pc]
    H["ey"] = causal_cal(H, "y_oo")
    C = route_costs(H, 6, c=0.3); H["ce"] = C["moc"][0]; H["cx"] = C["moo"][0]
    c = (H.ce + H.cx) / 1e4; sL = H.ey - 3 * c; sS = -H.ey - 3 * c - 0.01 / 252
    r, L = book(H, 6, sL, sS, H.y_oo, H.y_oo, H.ce + H.cx, H.ce + H.cx, ret_pos=True)
    m = metrics(r, 6); log(f"{pc} onml3-style k6 lam3 moo c0.3 (reproduction): $0.4M Sh {m['sh']:.2f} fold {m['fold']:.2f}")
    L = L.merge(H[["date", "tk"] + YS + ["n_hs0931", "n_hs0935", "n_hs1001", "cx"]], on=["date", "tk"], how="left")
    L["yr"] = L.date.dt.year
    t = decay_tab(L, f"{pc} k6 book 2017-26"); out.append(t)
    out.append(decay_tab(L[L.yr <= 2021], f"{pc} k6 2017-21")); out.append(decay_tab(L[L.yr >= 2022], f"{pc} k6 2022-26"))
    # extremes: top / bottom 5% of the day's prediction rank, all names
    E = H[(H.p >= 0.45) | (H.p < -0.45)].copy(); E["w"] = np.sign(E.p)
    out.append(decay_tab(E, f"{pc} extreme 5%"))
T = pd.concat(out); T.to_csv(f"{OUT}/step1_decay.csv", index=False)
tlog("STEP1 edge decay: gross bps per trade from the official close of t to each exit of t+1 (hsXXXX = mean predicted half-spread at the exit, t+1 model; moo_c03 = MOO cost per side at c=0.3, k6 order)", T)
log("STEP1 DONE")
