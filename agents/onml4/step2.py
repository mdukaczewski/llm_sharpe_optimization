#!/usr/bin/env python3
"""Step 2: exit and entry alternatives, each with its own predicted cost and its own cost-aware selection (k6, lam 3, both
prediction sets), per-name routing, MOO + 10:01 split, entry MOC / 15:59:30 cross / split.  Also the decay ratios used."""
from lib4 import *
G = load_G(); K = int(os.environ.get("K", "6")); LAM = 3.0
rows = []; tiers = []; mix = []
def run(H, pc, entry, exit, c, f, k=K, tag=""):
    r, L = route_book(H, k, LAM, entry=entry, exit=exit, c=c, f=f, ret_pos=True)
    nm = f"{pc} {entry}->{exit} c{c:g} f{f:g} k{k}{tag}"; m = metrics(r, k); row = mrow(nm, m)
    row["gross_bps"] = round((np.sign(L.w) * L.y).mean() * 1e4, 2); row["cost_bps"] = round(L.c.mean(), 2); row["xo"] = round(L.xo.mean(), 2)
    row["t1_pct"] = round((L.tier == 0).mean() * 100, 1); rows.append(row); log(str(row))
    if entry == "route" or exit == "route":
        mx = pd.concat([L.exit.value_counts(normalize=True).rename(lambda s: "exit_" + s), L.entry.value_counts(normalize=True).rename(lambda s: "entry_" + s)]).round(3)
        mx["book"] = nm; mix.append(mx)
        tb = L.groupby(["tier", "exit"]).size().unstack(fill_value=0); tb.index = [TNAMES[i] for i in tb.index]
        log(f"route mix by tier {nm}\n{tb.to_string()}")
    return r, L
for pc in ["p40", "p26"]:
    H = prep(G, pc)
    rt = H.groupby(["tier"])[[c for c in H.columns if c.startswith("rL_") or c.startswith("rS_")]].mean().round(3); rt.index = TNAMES[:len(rt)]
    log(f"{pc}: mean causal decay ratios (gross to exit / gross to open, extreme 5%) by tier\n{rt.to_string()}")
    for c in [0.1, 0.3, 1.0]: run(H, pc, "moc", "moo", c, 1.0)
    for x in ["x0935", "x1001", "x1031"]:
        for f in [1.0, 0.5]: run(H, pc, "moc", x, 0.3, f)
    for c, f in [(0.3, 1.0), (0.3, 0.5), (0.1, 1.0), (1.0, 1.0)]: run(H, pc, "moc", "split", c, f)
    for c, f in [(0.3, 1.0), (0.3, 0.5), (0.1, 1.0), (1.0, 1.0)]:
        r, L = run(H, pc, "moc", "route", c, f)
        if c == 0.3 and f == 1.0: t = tier_table(L); t.insert(0, "book", f"{pc} moc->route k{K}"); tiers.append(t)
    # entries (exit MOO c=0.3, then exit route)
    for f in [1.0, 0.5]: run(H, pc, "x1559", "moo", 0.3, f)
    run(H, pc, "split", "moo", 0.3, 1.0); run(H, pc, "route", "moo", 0.3, 1.0)
    for c, f in [(0.3, 1.0), (0.3, 0.5), (0.1, 1.0), (1.0, 1.0)]:
        r, L = run(H, pc, "route", "route", c, f)
        if c == 0.3 and f == 1.0: t = tier_table(L); t.insert(0, "book", f"{pc} route->route k{K}"); tiers.append(t)
    r, L = route_book(H, K, LAM, "moc", "moo", 0.3, 1.0, ret_pos=True); t = tier_table(L); t.insert(0, "book", f"{pc} moc->moo k{K}"); tiers.append(t)
    # entry-price diagnostic: gross from the 15:59 bar close instead of the official close
    L = L.merge(H[["date", "tk", "r_moc"]], on=["date", "tk"]); sg = np.sign(L.w)
    log(f"{pc} 15:59-bar entry diagnostic (moc->moo k{K} positions): side x (close/c389 - 1) mean {np.nanmean(sg*L.r_moc)*1e4:.2f} bps; by tier "
        + " ".join(f"{TNAMES[i]} {np.nanmean((sg*L.r_moc)[L.tier==i])*1e4:.2f}" for i in range(4)))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/step2_k{K}.csv", index=False)
tlog(f"STEP2 routes k{K} lam3 ($0.4M commission; gross/cost = bps per position, cost both sides; xo = extra orders per position)", T.drop(columns=["yrs"]))
tlog(f"STEP2 tier tables k{K}", pd.concat(tiers))
if mix: tlog(f"STEP2 route mix k{K}", pd.DataFrame(mix).fillna(0))
log("STEP2 DONE")
