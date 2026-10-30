#!/usr/bin/env python3
"""Step 3b: pR (rank average) with exit routing (route->route), and a smaller cAx weight (0.25), small grid; c-robustness of
the routes: selection at c_sel = 0.3, charged at c = 0.1 / 1.0."""
from lib4 import *
G = load_G()
def prank(s, d): return (s.groupby(d).rank(pct=True) - 0.5).astype(np.float32)
G["pR"] = prank((G.p40 + G.p26) / 2, G.date); G["pRX0.25"] = prank((G.p40 + G.p26) / 2 + 0.25 * 0.29 * G.zcAx.fillna(0), G.date)
rows = []
def rec(nm, r, k, L):
    m = metrics(r, k); row = mrow(nm, m); row["t1_pct"] = round((L.tier == 0).mean() * 100, 1); row["gross_bps"] = round((np.sign(L.w) * L.y).mean() * 1e4, 2)
    row["cost_bps"] = round(L.c.mean(), 2); rows.append(row); log(str({k_: v for k_, v in row.items() if k_ != "yrs"}))
H = prep(G, "pR")
for k in [6, 8]:
    for lam in [2.0, 3.0, 4.0]:
        r, L = route_book(H, k, lam, "route", "route", 0.3, 1.0, ret_pos=True); rec(f"pR route->route k{k} lam{lam:g}", r, k, L)
for ent, ex in [("route", "moo"), ("route", "route"), ("moc", "moo")]:
    for c in [0.1, 1.0]:
        r, L = route_book(H, 6, 3.0, ent, ex, c, 1.0, c_sel=0.3, ret_pos=True); rec(f"pR {ent}->{ex} k6 lam3 sel c0.3, charged c{c:g}", r, 6, L)
    for f in [0.5, 2.0]:
        r, L = route_book(H, 6, 3.0, ent, ex, 0.3, f, f_sel=1.0, ret_pos=True); rec(f"pR {ent}->{ex} k6 lam3 sel f1, charged f{f:g}", r, 6, L)
H = prep(G, "pRX0.25")
for k in [6, 8]:
    for lam in [2.0, 3.0]:
        r, L = route_book(H, k, lam, "route", "moo", 0.3, 1.0, ret_pos=True); rec(f"pRX0.25 route->moo k{k} lam{lam:g}", r, k, L)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/step3b.csv", index=False)
tlog("STEP3b pR route->route grid, c/f robustness (selection fixed at c=0.3 / f=1), cAx weight 0.25", T.drop(columns=["yrs"]))
log("STEP3b DONE")
