#!/usr/bin/env python3
"""Steps 1-2: cAx books on universe(150/300/600) with the official-open exit (MOC + MOO, auction cost at c = 0.1/0.3/1.0),
the 09:31-bar exit crossing the predicted half-spread (x1) and paying half of it (x05), and the old 1 bp convention.
Output: out/s1_table.csv (one row per book x universe x execution), tier tables for the 150-name books, log.txt."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3lib import *
log(f"s1_books start DEV_END={DEV_END}")
D = load_frame(600); sig = cax(D)
BOOKS = [(10, "lo", 1.5, "lo10_s15"), (10, "lo", 2.0, "lo10_s2"), (20, "lo", 1.5, "lo20_s15"), (20, "lo", 2.0, "lo20_s2"), (10, "ls", None, "ls10"), (20, "ls", None, "ls20")]
EXEC = [("moo_c03", dict(exit="oo", c=0.3)), ("moo_c01", dict(exit="oo", c=0.1)), ("moo_c1", dict(exit="oo", c=1.0)),
        ("x1", dict(exit="on", f=1.0, c=0.3)), ("x05", dict(exit="on", f=0.5, c=0.3)), ("old1bp", dict(old=True))]
rows = []; series = {}
for u in [150, 300, 600]:
    sub = D["rank"] <= u
    for k, mode, smin, bk in BOOKS:
        P = positions(D[sub], sig[sub], k=k, mode=mode, smin=smin); npos = k * (2 if mode == "ls" else 1)
        log(f"--- {bk} u{u}: picks/day {len(P)/P.date.nunique():.1f} days {P.date.nunique()} mean gross y_oo {np.mean(np.sign(P.w)*P.y_oo)*1e4:.1f} bps y_on {np.mean(np.sign(P.w)*P.y_on)*1e4:.1f} bps | mean cost/side c=0.3: in {cost_sides(P,0.3)[0].mean():.2f} out {cost_sides(P,0.3)[1].mean():.2f} | hs0931 mean {P.hs0931.mean():.1f} median {P.hs0931.median():.1f}")
        for ex, kw in EXEC:
            r = net(P, **kw); name = f"{bk}_u{u}_{ex}"; d = rep(r, name, npos); d.update(book=bk, u=u, exe=ex, npos=npos, picks=len(P) / P.date.nunique())
            d.update(corrs(r)); rows.append(d); series[name] = r
        if u == 150 or (bk in ("lo10_s15", "ls10")):
            T = tier_table(P, 0.3, "oo"); log(f"tiers (y_oo, c=0.3) {bk} u{u}:\n" + T.round(2).to_string(index=False))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/s1_table.csv", index=False)
pd.DataFrame(series).to_parquet(f"{OUT}/s1_series.parquet")
log("summary (Sh / fold-OOS at the $3 commission), rows = book x universe, columns = execution:")
for col in ["sh", "oos"]:
    piv = T.pivot_table(index=["book", "u"], columns="exe", values=col).reindex(columns=[e for e, _ in EXEC]).round(2)
    log(f"{col}:\n{piv.to_string()}")
# family fold-OOS over the moo_c03 books per universe (the chooser's view)
for u in [150, 300, 600]:
    fam = {n: s for n, s in series.items() if n.endswith(f"_u{u}_moo_c03")}
    oos, tab = fold_oos({n: s - comm_bps(T.set_index("name").npos[n]) / 1e4 for n, s in fam.items()}, p=False)
    log(f"family fold-OOS moo_c03 u{u}: {sh(oos):.2f} picks {tab.pick.value_counts().to_dict()}")
log("s1_books done")
