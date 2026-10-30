#!/usr/bin/env python3
"""Step 3: liquidity as a feature of the rule (official-open exit, c = 0.3, $3 commission).
(a) filters: skip names whose predicted hs0931 > cap, or whose auction cost per side at c = 0.3 (max of entry / exit at the
    book's order size) > cap, cap in 5 / 10 / 20 bps, applied BEFORE ranking (the book picks from the remaining names).
(b) a liquidity term in the composite with a FIXED weight: cAx - w x rank(hs0931) or cAx - w x rank(hl1), w in 0.3 / 0.5
    (cross-sectional pct rank, demeaned, inside the book's universe); 3-value check only, no optimisation."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3lib import *
log("s2_liqfeat start")
D = load_frame(600); sig0 = cax(D)
BOOKS = [(10, "lo", 1.5, "lo10_s15"), (20, "lo", 1.5, "lo20_s15"), (10, "ls", None, "ls10"), (20, "ls", None, "ls20")]
rows = []; series = {}
def run(sub, s, k, mode, smin, name, filt=None):
    P = positions(D[sub], s[sub], k=k, mode=mode, smin=smin, filt=None if filt is None else filt[sub]); npos = k * (2 if mode == "ls" else 1)
    r = net(P, exit="oo", c=0.3); d = rep(r, name, npos); d.update(npos=npos, picks=len(P) / P.date.nunique(), hs=P.hs0931.mean(), cost=(cost_sides(P, 0.3)[0] + cost_sides(P, 0.3)[1]).mean()); d.update(corrs(r)); rows.append(d); series[name] = r; return r
for u in [150, 300, 600]:
    sub = D["rank"] <= u
    rhs = (D[sub].groupby("date").hs0931.rank(pct=True) - 0.5).reindex(D.index)          # ranks inside the book universe
    rhl = (D[sub].groupby("date").hl1.rank(pct=True) - 0.5).reindex(D.index)
    for k, mode, smin, bk in BOOKS:
        if u == 600 and bk in ("lo20_s15", "ls20"): pass
        npos = k * (2 if mode == "ls" else 1); order = CAP / npos; rv = D.rv20 / np.sqrt(252)
        ac = np.maximum(liq.auction_cost_bps(order, np.maximum(np.exp(D.dvc), 0.05 * np.exp(D.ldv20)), rv, 0.3), liq.auction_cost_bps(order, np.exp(D.dv0), rv, 0.3))
        ac = pd.Series(ac, index=D.index)
        log(f"=== {bk} u{u}")
        run(sub, sig0, k, mode, smin, f"{bk}_u{u}_base")
        for cap in [5, 10, 20]:
            f = (D.hs0931 <= cap); log(f"  filter hs0931<={cap}: keeps {f[sub].mean()*100:.0f}% of rows")
            run(sub, sig0, k, mode, smin, f"{bk}_u{u}_fhs{cap}", filt=f)
            f = (ac <= cap); log(f"  filter auction<={cap}: keeps {f[sub].mean()*100:.0f}% of rows")
            run(sub, sig0, k, mode, smin, f"{bk}_u{u}_fac{cap}", filt=f)
        for w in [0.3, 0.5]:
            run(sub, sig0 - w * rhs.fillna(0), k, mode, smin, f"{bk}_u{u}_rhs{w}")
            run(sub, sig0 - w * rhl.fillna(0), k, mode, smin, f"{bk}_u{u}_rhl{w}")
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/s2_table.csv", index=False); pd.DataFrame(series).to_parquet(f"{OUT}/s2_series.parquet")
T["variant"] = T.name.str.split("_u").str[1].str.split("_", n=1).str[1]; T["bu"] = T.name.str.rsplit("_", n=1).str[0]
for col in ["oos", "sh"]:
    log(f"{col} (at commission), rows = book_universe, columns = variant:\n" + T.pivot_table(index="bu", columns="variant", values=col).round(2).to_string())
log("s2_liqfeat done")
